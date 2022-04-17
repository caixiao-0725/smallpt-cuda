#include<algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <algorithm>
#include <climits>
#include <chrono>
#include "helper_math.h"

#define WIDTH 1024
#define HEIGHT 768
#define SPP 4096
#define BOUNCE 10
#define SPHERE_EPSILON 0.0001f
#define RAY_EPSILON 0.02f;
#define M_PI 3.1415926

struct Ray
{
    __device__ Ray(float3 pos, float3 dir) :
        pos(pos), dir(dir) {}

    float3 pos;
    float3 dir;
};

enum class Material { Diffuse, Specular, Refractive };

struct Sphere
{
    __device__ float intersect(const Ray& ray) const
    {
        float t;
        float3 dis = pos - ray.pos;
        float proj = dot(dis, ray.dir);
        float delta = proj * proj - dot(dis, dis) + radius * radius;

        if (delta < 0) return 0;

        delta = sqrtf(delta);
        return (t = proj - delta) > SPHERE_EPSILON ? t : ((t = proj + delta) > SPHERE_EPSILON ? t : 0);
    }

    float radius;
    float3 pos, emissionColor, mainColor;
    Material material;
};

__constant__ Sphere spheres[] =
{
    { 1e5f, { 1e5 + 1,40.8,81.6 }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, Material::Diffuse }, // Left 
    { 1e5f, { -1e5 + 99,40.8,81.6 }, { 0.0f, 0.0f, 0.0f }, { 0.25f, 0.25f, 0.75f }, Material::Diffuse }, // Right 
    { 1e5f, {50,40.8, 1e5 }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.75f, 0.75f }, Material::Diffuse }, // Back 
    { 1e5f, { 50,40.8,-1e5 + 170 }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, Material::Diffuse }, // Front 
    { 1e5f, { 50, 1e5, 81.6 }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.75f, 0.75f }, Material::Diffuse }, // Bottom 
    { 1e5f, { 50,-1e5 + 81.6,81.6 }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.75f, 0.75f }, Material::Diffuse }, // Top 
    { 16.5f, { 27,16.5,47 }, { 0.0f, 0.0f, 0.0f }, { 0.999f, 0.999f, 0.999f }, Material::Specular }, // small sphere 1
    { 16.5f, { 73,16.5,78 }, { 0.0f, 0.0f, 0.0f }, { 0.999f, 0.999f, 0.999f }, Material::Refractive }, // small sphere 2
    { 600.0f, { 50,681.6 - .27,81.6 }, { 10.0f,10.0f,10.0f }, { 0.0f, 0.0f, 0.0f }, Material::Diffuse }  // Light
};

__device__ inline bool intersectScene(const Ray& ray, float& t, int& id)
{
    t = FLT_MAX, id = -1;
    int sphereNum = sizeof(spheres) / sizeof(Sphere);
    for (int i = 0; i < sphereNum; i++)
    {
        float ct = spheres[i].intersect(ray);
        if (ct != 0 && ct < t)
        {
            t = ct;
            id = i;
        }
    }

    return id != -1;
}

__device__ static float rand(uint* seed0, uint* seed1)
{
    *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);
    *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

    uint ires = ((*seed0) << 16) + (*seed1);

    union
    {
        float f;
        uint ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

    return (res.f - 2.f) / 2.f;
}

inline int gammaCorrect(float c)
{
    return int(pow(clamp(c, 0.0f, 1.0f), 1 / 2.2) * 255 + .5);
}

__device__ float3 pathTrace(Ray& ray, uint* s1, uint* s2)
{
    float3 accumColor = make_float3(0.0f, 0.0f, 0.0f);
    float3 mask = make_float3(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < BOUNCE; i++)
    {
        float t;
        int id;

        if (!intersectScene(ray, t, id))
            return make_float3(0.0f, 0.0f, 0.0f);

        const Sphere& obj = spheres[id];
        float3 p = ray.pos + ray.dir * t;
        float3 n = normalize(p - obj.pos);  //n始终是向球体外的
        float3 nl = dot(n, ray.dir) < 0 ? n : n * -1;  //从外向内还是n

        accumColor += mask * obj.emissionColor;
        if (obj.material == Material::Diffuse) {
            float r1 = rand(s1, s2) * M_PI * 2;
            float r2 = rand(s1, s2);
            float r2s = sqrtf(r2);

            float3 w = nl;
            float3 u = normalize(cross((std::fabs(w.x) > std::fabs(w.y) ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
            float3 v = cross(w, u);

            float3 d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));

            ray.pos = p + nl * RAY_EPSILON;
            ray.dir = d;

            mask *= obj.mainColor * dot(d, nl) * 2;
        }
        else if (obj.material == Material::Specular) {
            ray.pos = p + nl * RAY_EPSILON;
            ray.dir = ray.dir - 2 * n * dot(n, ray.dir);
            mask *= obj.mainColor;
        }
        else if (obj.material == Material::Refractive) {
            
            float3 reflect_dir = ray.dir - 2 * n * dot(n, ray.dir);
            bool into = dot(n,nl) > 0;

            float nc = 1;
            float nt = 1.5;
            float nnt = into ? nc / nt : nt / nc;
            float ddn = dot(nl,ray.dir), cos2t;
            cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
            if ( cos2t< 0) {
                ray.pos = p + nl * RAY_EPSILON;
                ray.dir = reflect_dir;
                mask *= obj.mainColor;
            }
            else {
                float3 tdir = normalize(ray.dir * nnt - n  * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t))));
                float a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : dot(n, tdir));
                float Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
                float r2 = rand(s1, s2);

                if (r2 < P) {
                    if (into) {
                        ray.pos = p + n * RAY_EPSILON;
                    }
                    else
                    {
                        ray.pos = p - n * RAY_EPSILON;
                    }
                    ray.dir = reflect_dir;
                    mask *= obj.mainColor * RP;
                }
                else {
                    if (into) {
                        ray.pos = p - n * RAY_EPSILON;
                    }
                    else
                    {
                        ray.pos = p + n * RAY_EPSILON;
                    }
                    ray.dir = tdir;
                    mask *= obj.mainColor * TP;
                }
            }
        }

    }

    return accumColor;
}

__global__ void pathTracekernel(float3* h_output)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    uint i = (HEIGHT - y - 1) * WIDTH + x;

    uint s1 = x;
    uint s2 = y;
    Ray camRay(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1)));
    float3 cx = make_float3(WIDTH * 0.5135 / HEIGHT, 0.0f, 0.0f);
    float3 cy = normalize(cross(cx, camRay.dir)) * 0.5135;
    float3 pixel = make_float3(0.0f);

    for (int s = 0; s < SPP; s++)
    {
        float3 d = camRay.dir + cx * ((x+rand(&s1,&s2)-0.5) / WIDTH - 0.5) + cy * ((y + rand(&s1, &s2) - 0.5) / HEIGHT - 0.5);
        Ray ray(camRay.pos + d * 140, normalize(d));

        pixel += pathTrace(ray, &s1, &s2) * (1. / SPP);
    }

    h_output[i] = clamp(pixel, 0.0f, 1.0f);
}

int main() {

    float3* h_output = new float3[WIDTH * HEIGHT];
    float3* d_output;

    cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(float3));

    dim3 block(8, 8, 1);
    dim3 grid(WIDTH / block.x, HEIGHT / block.y, 1);

    printf("CUDA initialized.\nStart rendering...\n");

    std::chrono::time_point<std::chrono::system_clock> begin = std::chrono::system_clock::now();

    pathTracekernel << <grid, block >> > (d_output);

    cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * sizeof(float3), cudaMemcpyDeviceToHost);

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedTime = end - begin;
    printf("Time: %.6lfs\n", elapsedTime.count());

    cudaFree(d_output);

    printf("Done!\n");

    FILE* f = fopen("smallptcuda1.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", WIDTH, HEIGHT, 255);
    for (int i = 0; i < WIDTH * HEIGHT; i++)
        fprintf(f, "%d %d %d ", gammaCorrect(h_output[i].x),
            gammaCorrect(h_output[i].y),
            gammaCorrect(h_output[i].z));

    printf("Saved image to 'smallptcuda.ppm'.\n");

    delete[] h_output;

    return 0;
}
