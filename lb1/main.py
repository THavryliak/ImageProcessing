import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

low_contrast_image = 'stones.jpg'
save_filename = 'output.jpg'
img = Image.open(low_contrast_image)
image_arr = np.asarray(Image.open(low_contrast_image))


def validate_rgb_value(rgb_val: int):
    if rgb_val > 0:
        return min(255, int(rgb_val))
    else:
        return 0


def contrast(image_arr, a: float, s: int, t: int):
    width, height, z = image_arr.shape
    for i in range(0, width):
        for j in range(0, height):
            rgb = image_arr[i][j]
            new_r = validate_rgb_value(a * (rgb[0] + s) - t)
            new_g = validate_rgb_value(a * (rgb[1] + s) - t)
            new_b = validate_rgb_value(a * (rgb[2] + s) - t)
            image_arr[i][j] = [new_r, new_g, new_b]
    return image_arr


updated_low_contrast_image = contrast(image_arr, 0.8, 30, 3)


eq_img = Image.fromarray(updated_low_contrast_image)
eq_img.save(save_filename)

fig = plt.figure()
fig.set_figheight(20)
fig.set_figwidth(20)

# display old image
fig.add_subplot(1, 2, 1)
plt.imshow(img)

# display the new image
fig.add_subplot(1, 2, 2)
plt.imshow(eq_img)

plt.show(block=True)

