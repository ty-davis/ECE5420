def main():
    num = input("What is the last number of your W number? ")
    with open(f"source{num}.txt", 'r') as fin:
        source = fin.read()
    if not source:
        print("Invalid input")
        return

    # test data
    # source = "1111111111111101111001111001111100"
    # source = "01000000000000000000000111111111111111000000000000000000000000111111111111111111111111111111000000000000000000000000000000000000000000000000000000000000"

    code = lz(source)

    print("FINAL CODE:")
    print("as list:", code)
    result = display_code(code)
    print("as data:", result)
    print(len(result), "chars long")

    with open('codewords_01367741.txt', 'w') as fout:
        print("Writing to file...")
        fout.write(result)
        print("Done.")


def lz(source):
    d1 = { '': 0 }
    max_dict_location = 0
    ptr = 0
    code = []

    while ptr < len(source) and ptr >= 0:
        last_section = ''
        for e in range(len(source) - ptr):
            section = source[ptr:ptr+e]
            if not section in d1.keys():
                max_dict_location += 1
                d1[section] = max_dict_location
                code.append((d1[last_section], section[-1]))
                ptr += e
                break
            elif ptr + e == len(source) - 1:
                if section:
                    code.append((d1[section], section[-1]))
                ptr = -1000
                break
            last_section = section
    return code

def display_code(code):
    max_code_loc = max(x[0] for x in code)
    code_len = len(bin(max_code_loc)[2:])
    r = ''
    for loc, c in code:
        b = bin(loc)[2:]
        r += "0" * (code_len - len(b)) + b + c
    return r


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
