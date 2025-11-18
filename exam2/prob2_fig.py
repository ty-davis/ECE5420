import matplotlib.pyplot as plt


def main():
    plt.figure(figsize=(6, 3))
    ax = plt.gca()
    plt.scatter(-4, 0, color='black')
    plt.scatter(0, 0, color='black')
    plt.scatter(2, 0, color='black')
    plt.scatter(4, 0, color='black')

    # Clean up the axes for exam appearance
    plt.axis('equal')
    plt.xlim(-4.5, 4.5)
    plt.ylim(-2, 2)
    
    # Remove the frame/border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Remove ticks and labels
    plt.xticks(ax.get_xticks()[1::2])
    plt.yticks(ax.get_xticks()[1::2])

    ax.axhline(y=0, color='gray')
    ax.axvline(x=-2, color='gray', linestyle='--')
    ax.axvline(x=1, color='gray', linestyle='--')
    ax.annotate('1', (1.1, -1))
    ax.axvline(x=3, color='gray', linestyle='--')
    ax.annotate('3', (3.1, -1))

    
    # Optional: keep a subtle grid
    plt.grid()
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
