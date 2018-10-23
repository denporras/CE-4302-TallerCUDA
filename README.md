# Evaluación Taller 4: CUDA
Taller del curso Arquitectura de Computadores II, donde se utilizan GPUs NVIDIA utilizando CUDA.

# Instrucciones
Para ejecutar estos programas se debe conectar a las máquinas con GPU del SipLab.

## Multiplicación
1- Compilar con el siguiente comando: 
  ```console
  nvidia@SIPLab:~$ nvcc matrix_mult.cu -o matrix_mult

  ```
2- Ejecutar el programa:
  ```console
  nvidia@SIPLab:~$ ./matrix_mult

  ```
## Convolución
1- Compilar con el siguiente comando: 
  ```console
  nvidia@SIPLab:~$ nvcc convolution.cu -o convolution

  ```
2- Ejecutar el programa:
  ```console
  nvidia@SIPLab:~$ ./convolution

  ```