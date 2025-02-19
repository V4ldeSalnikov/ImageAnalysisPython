import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt


def histEqualization(channel: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to a single channel.

    Parameters:
        channel (np.ndarray): Grayscale image channel.

    Returns:
        np.ndarray: Equalized channel.
    """
    hist, _ = np.histogram(channel.flatten(), 256, [0, 255])
    cdf = hist.cumsum()
    cdf_min = cdf.min()
    cdf_max = cdf.max()

    cdf_norm = (cdf - cdf_min) * 255 / (cdf_max - cdf_min) if (cdf_max - cdf_min) != 0 else cdf
    channel_equalized = cdf_norm[channel.flatten().astype(np.int32)]
    channel_equalized = np.reshape(channel_equalized, channel.shape)
    return channel_equalized.astype(np.uint8)


def main():
    st.title("Histogram Equalization")
    st.sidebar.header("Settings")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        try:
            image = cv2.imread("Unequalized.jpg")
            if image is None:
                st.error("Default image 'Unequalized.jpg' not found. Please upload an image.")
                return
            st.sidebar.info("Using default image: Unequalized.jpg. You can change it by uploading a file.")
        except Exception as e:
            st.error(f"Error loading default image: {e}")
            return
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[..., 0] = histEqualization(img_yuv[..., 0])
    equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Original Image", use_container_width=True)
    with col2:
        st.image(equalized_rgb, caption="Histogram Equalized Image", use_container_width=True)

    input_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    output_yuv = cv2.cvtColor(equalized, cv2.COLOR_BGR2YUV)
    input_y = input_yuv[..., 0]
    output_y = output_yuv[..., 0]

    input_hist = cv2.calcHist([input_y], [0], None, [256], [0, 256])
    output_hist = cv2.calcHist([output_y], [0], None, [256], [0, 256])

    x_vals = np.arange(256)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].bar(x_vals, input_hist.ravel(), color='blue', width=1)
    ax[0].set_title("Original Image Histogram (Y channel)")
    ax[0].set_xlim([0, 256])
    ax[0].set_xlabel("Intensity")
    ax[0].set_ylabel("Frequency")

    ax[1].bar(x_vals, output_hist.ravel(), color='green', width=1)
    ax[1].set_title("Equalized Image Histogram (Y channel)")
    ax[1].set_xlim([0, 256])
    ax[1].set_xlabel("Intensity")
    ax[1].set_ylabel("Frequency")

    st.pyplot(fig)


if __name__ == "__main__":
    main()
