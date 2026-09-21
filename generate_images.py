import os
import struct
import random

INPUT_DIR = "input"

NUM_IMAGES = 120
WIDTH = 512
HEIGHT = 512


def create_image(filename, seed):
    random.seed(seed)

    with open(filename, "wb") as f:

        # PPM header
        f.write(b"P6\n")
        f.write(f"{WIDTH} {HEIGHT}\n".encode())
        f.write(b"255\n")

        for y in range(HEIGHT):

            for x in range(WIDTH):

                # Create a structured image pattern
                r = int(
                    127
                    + 127
                    * ((x + seed * 3) % WIDTH)
                    / WIDTH
                )

                g = int(
                    127
                    + 127
                    * ((y + seed * 5) % HEIGHT)
                    / HEIGHT
                )

                b = int(
                    127
                    + 127
                    * (
                        ((x + y + seed * 7)
                         % (WIDTH + HEIGHT))
                        / (WIDTH + HEIGHT)
                    )
                )

                # Add deterministic noise
                noise = random.randint(-20, 20)

                r = max(0, min(255, r + noise))
                g = max(0, min(255, g + noise))
                b = max(0, min(255, b + noise))

                f.write(
                    struct.pack(
                        "BBB",
                        r,
                        g,
                        b
                    )
                )


def main():

    os.makedirs(INPUT_DIR, exist_ok=True)

    print("Generating CUDA image-processing dataset...")
    print(f"Number of images: {NUM_IMAGES}")
    print(f"Resolution: {WIDTH} x {HEIGHT}")

    for i in range(NUM_IMAGES):

        filename = os.path.join(
            INPUT_DIR,
            f"image_{i:03d}.ppm"
        )

        create_image(filename, i)

        if (i + 1) % 10 == 0:
            print(
                f"Generated {i + 1}/{NUM_IMAGES} images"
            )

    print()
    print("Dataset generation complete.")
    print(f"Images saved in: {INPUT_DIR}/")


if __name__ == "__main__":
    main()
