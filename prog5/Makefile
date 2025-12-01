CC = gcc
CFLAGS = -fPIC -Wall -I./include
LDFLAGS = -shared

SRC = channel_coding.c noise.c
OBJ = $(SRC:.c=.o)
TARGET = channel_coding.so

all: $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(OBJ) $(TARGET)
