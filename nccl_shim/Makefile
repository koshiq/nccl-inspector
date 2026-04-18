CC      = gcc
NVCC    = /opt/cuda/bin/nvcc
CFLAGS  = -O2 -fPIC -shared -Wall -I/usr/include -I/opt/cuda/include
LDFLAGS = -ldl -lrt -lnccl

TARGET  = libnccl_shim.so
TEST    = test_harness
TEST_MP = test_harness_mp

all: $(TARGET) $(TEST) $(TEST_MP)

$(TARGET): shim.c
	$(CC) $(CFLAGS) -o $(TARGET) shim.c $(LDFLAGS)

$(TEST): test_harness.cu
	$(NVCC) -I/usr/include -L/usr/lib -o $(TEST) test_harness.cu -lnccl -lcudart

$(TEST_MP): test_harness_mp.cu
	$(NVCC) -I/usr/include -L/usr/lib -o $(TEST_MP) test_harness_mp.cu -lnccl -lcudart

clean:
	rm -f $(TARGET) $(TEST) $(TEST_MP)
