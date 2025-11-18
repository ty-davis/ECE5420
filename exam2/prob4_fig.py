import matplotlib.pyplot as plt

def gray_code(idx):
    hb = idx % 4
    lb = idx // 4
    return f"{lb^(lb >> 1):02b}{hb ^ (hb >> 1):02b}"

def main():
    plt.figure(figsize=(6, 6))
    for j in range(0, 4):
        for i in range(0, 4):
            x = i * 2 - 3
            y = -j * 2 + 3
            idx = j * 4 + i
            bits = gray_code(idx)
            plt.annotate(f"{bits}", (x, y+0.1))
            plt.scatter(x,y, marker='o', c='black')

    # Clean up the axes for exam appearance
    plt.axis('equal')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    
    # Remove the frame/border
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    
    # Remove ticks and labels
    plt.xticks(plt.gca().get_xticks(), [f"{int(x)}A" if int(x) != 0 else '0' for x in plt.gca().get_xticks()])
    plt.yticks(plt.gca().get_xticks(), [f"{int(y)}A" if int(y) != 0 else '0' for y in plt.gca().get_yticks()])
    
    # Optional: keep a subtle grid
    plt.grid()
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
