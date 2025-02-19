import streamlit as st
import numpy as np
from PIL import Image, ImageOps


def convolve(image, kernel):
    """
    Apply a convolution filter to a grayscale image.

    Parameters:
        image (PIL.Image): Grayscale image.
        kernel (np.ndarray): Convolution kernel.

    Returns:
        PIL.Image: Convolved image.
    """
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    img_array = np.array(image, dtype=np.float32)
    padded_image = np.pad(img_array, pad_size, mode='edge')
    output_array = np.zeros_like(img_array)

    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            region = padded_image[y:y + kernel_size, x:x + kernel_size]
            output_array[y, x] = np.sum(region * kernel)

    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
    return Image.fromarray(output_array)


def main():
    st.title("Image Convolution")
    st.sidebar.header("Settings")

    # kernels for 3x3 sizes.
    kernel_options = {
        "Identity": np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]]),
        "Smoothing (3x3)": np.array([[1, 1, 1],
                                     [1, 1, 1],
                                     [1, 1, 1]]) / 9,
        "Sharpen (3x3)": np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]]),
        "Edge Detection (Horizontal)": np.array([[-1, -2, -1],
                                                 [0, 0, 0],
                                                 [1, 2, 1]]),
        "Edge Detection (Vertical)": np.array([[-1, 0, 1],
                                               [-2, 0, 2],
                                               [-1, 0, 1]]),
        "Shift Right": np.array([[0, 0, 0],
                                 [0, 0, 1],
                                 [0, 0, 0]]),
    }

    selected_kernel = st.sidebar.selectbox("Select Kernel:", list(kernel_options.keys()))
    kernel = kernel_options[selected_kernel]
    kernel_size = st.sidebar.selectbox("Kernel Size:", [3, 5, 7], index=0)

    # Adjust the kernel if the selected kernel size is different from 3.
    if kernel.shape[0] != kernel_size:
        if selected_kernel == "Smoothing (3x3)":
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        elif selected_kernel == "Sharpen (3x3)":
            center_value = kernel_size * kernel_size - 1
            kernel = -np.ones((kernel_size, kernel_size))
            kernel[kernel_size // 2, kernel_size // 2] = center_value
        elif selected_kernel == "Edge Detection (Horizontal)":
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[0, :] = -1
            kernel[-1, :] = 1
            kernel[0, kernel_size // 2] = -2
            kernel[-1, kernel_size // 2] = 2
        elif selected_kernel == "Edge Detection (Vertical)":
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[:, 0] = -1
            kernel[:, -1] = 1
            kernel[kernel_size // 2, 0] = -2
            kernel[kernel_size // 2, -1] = 2
        elif selected_kernel == "Shift Right":
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, -1] = 1
        else:  # Identity kernel
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, kernel_size // 2] = 1

    st.sidebar.write("Kernel Matrix:")
    st.sidebar.write(kernel)

    #  filter descriptions
    filter_descriptions = {
        "Identity": r"""**Identity Filter:** This filter leaves the image unchanged.

$$ I' = I $$""",
        "Smoothing (3x3)": r"""**Smoothing Filter:** This filter averages the pixel values in the neighborhood.

$$$$""",
        "Sharpen (3x3)": r"""**Sharpen Filter:** This filter enhances edges by emphasizing differences.

$$ I'(x,y) = 5I(x,y) - I(x-1,y) - I(x+1,y) - I(x,y-1) - I(x,y+1) $$""",
        "Edge Detection (Horizontal)": r"""**Edge Detection (Horizontal):** Detects horizontal edges.

$$ K = \begin{pmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{pmatrix} $$""",
        "Edge Detection (Vertical)": r"""**Edge Detection (Vertical):** Detects vertical edges.

$$ K = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix} $$""",
        "Shift Right": r"""**Shift Right Filter:** Shifts the image to the right.

$$ I'(x,y) = I(x-1, y) $$""",
    }

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        image = Image.open("lund.jpeg")
        st.sidebar.info("Using default image: lund.jpeg. You can change it by browsing files.")

    else:
        image = Image.open(uploaded_file)

    image = ImageOps.grayscale(image)
    image = image.resize((200, 200))

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        convolved_image = convolve(image, kernel)
        st.image(convolved_image, caption="Convolved Image", use_container_width=True)

    st.markdown(filter_descriptions.get(selected_kernel, ""))


if __name__ == "__main__":
    main()
