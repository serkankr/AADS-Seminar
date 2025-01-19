import random

file_sizes = [10, 15, 20, 25, 50]

random_numbers = random.sample(range(1, 100), 50)

for size in file_sizes:
    file_name = f"data_{size}.txt"
    with open(file_name, 'w') as f:
        for num in random_numbers[:size]:
            f.write(f"{num}\n")

file_names = [f"data_{size}.txt" for size in file_sizes]
file_names