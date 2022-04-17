# use cuda to finish smallpt
使用cuda去实现smallpt  
![test](https://github.com/caixiao-0725/smallpt-cuda/blob/main/images/smallpt_antialised.png)   
samples设置为4096，bounce设置为10，按照cpu的速度要124min，但是在gpu上只用了224s(RTX 3060)。  
坑:__constant__的使用过程中，使用了析构函数，导致报错。析构函数在调用这个class的时候就会自动赋值，这和不变的constant相违背。    
下周争取完成Games103的课后作业  
