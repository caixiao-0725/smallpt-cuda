# use cuda to finish smallpt
Use cuda to finish smallpt  
使用cuda去实现smallpt  

坑:__constant__的使用过程中，使用了析构函数，导致报错。析构函数在调用这个class的时候就会自动赋值，这和不变的constant相违背。  
