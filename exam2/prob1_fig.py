import matplotlib.pyplot as plt
import numpy as np

def main():
    plt.figure(figsize=(6, 6))
    ax = plt.gca()



    # Clean up the axes for exam appearance
    plt.axis('equal')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    
    # Remove the frame/border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Draw custom grid lines
    ax.axhline(y=np.sqrt(3), color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
    ax.axhline(y=-np.sqrt(3), color='gray', linestyle='--', alpha=0.7, linewidth=0.8)

    # Also add vertical grid lines if needed
    for x in ax.get_xticks():
        ax.axvline(x=x, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

    # Optional: keep a subtle grid
    plt.grid()
    
    # top and bottom lines of dots
    for j in range(2):
        y = (j*2 - 1) * np.sqrt(3)
        for i in range(3):
            x = (i*2)-2
            plt.scatter(x, y, c='black')

    # middle line
    for i in range(2):
        x = i*2 - 1
        y = 0
        plt.scatter(x, y, c='black')
    
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
