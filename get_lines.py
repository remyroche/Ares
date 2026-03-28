import sys

def main():
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
        lines = f.readlines()
        for i in range(start-1, end):
            print(f"{i+1}: {lines[i]}", end='')

if __name__ == '__main__':
    main()
