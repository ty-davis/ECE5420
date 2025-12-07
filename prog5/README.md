# Program 5 - Channel Coding

![BER Figure](doc/ber_fig.png)

## Running the code

The c files have been compiled for use on Windows and Linux. With the correct
shared library file (.dll for Windows and .so for Linux) in the directory,
simply run:

```bash
python3 ChannelCoding.py
```

## Configuring the simulation

The number of bits to run for each selected SNR value is specified
in a Python dictionary at the top of the ChannelCoding.py file. It
looks like this:

```python
snr_bitcounts = {
    0: 10**7,
    1: 10**7,
    2: 10**7,
    3: 10**7,
    4: 10**7,
    5: 10**8,
    6: 10**8,
    7: 10**8,
    8: 10**8,
    9: 10**9,
    10: 10**9,
    11: 10**10,
    12: 4 * 10**10,
}
```

Modifying this dict (removing rows or changing keys and values) will
allow you to run any number of SNR values or configure how many bits
to simulate at each one.

## Building from source

Use `make` to compile the shared library binaries for either Windows or Linux.

Linux:
```bash
make
```

Windows:
```bash
make windows
```

