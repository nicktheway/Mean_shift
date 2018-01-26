CC=nvcc
CFLAGS=-O3 -lm

%.o: %.cu
	$(CC) -c -o $@ $^ $(CFLAGS)
	
all: ms_global.o ms_shared.o
	$(CC) -o glms $< $(CFLAGS)
	$(CC) -o shms $(filter-out $<,$^) $(CFLAGS)

seeds:
	$(CC) ms_global.cu -o sglms $(CFLAGS) -D SEED_DATASET
	$(CC) ms_shared.cu -o sshms $(CFLAGS) -D SEED_DATASET
	
clean:
	rm *.o
