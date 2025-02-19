import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt



def create_test_image():
    img = np.zeros((200, 300), dtype=np.uint8)

    cv2.putText(img, 'ABC', (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255), 5, cv2.LINE_AA)

    noise_ratio = 0.02
    num_noisy = int(noise_ratio * img.size)

    # Randomly choose pixel locations
    coords = [np.random.randint(0, i - 1, num_noisy) for i in img.shape]

    half = num_noisy // 2
    img[coords[0][:half], coords[1][:half]] = 255
    img[coords[0][half:], coords[1][half:]] = 0

    return img



operation_overview = {
    "Erosion": (
        "**Erosion** shrinks the white regions in the image. "
        "It removes boundary pixels, making objects smaller and "
        "can help remove small white noise."
    ),
    "Dilation": (
        "**Dilation** expands the white regions in the image. "
        "It adds boundary pixels, making objects bigger and "
        "can help fill small holes in white objects."
    ),
    "Opening": (
        "**Opening** = Erosion followed by Dilation. "
        "It removes small white noise while preserving the overall shape."
    ),
    "Closing": (
        "**Closing** = Dilation followed by Erosion. "
        "It fills small black holes/gaps inside white objects."
    ),
}


def main():
    st.title("Morphological Operations")
    st.write("""
        This app demonstrates **erosion**, **dilation**, **opening**, and **closing**
        using different structuring elements (rectangle, cross, ellipse).
        You can use the default test image (white "ABC" with noise) or upload your own.
    """)

    operation_dict = {
        "Erosion": cv2.MORPH_ERODE,
        "Dilation": cv2.MORPH_DILATE,
        "Opening": cv2.MORPH_OPEN,
        "Closing": cv2.MORPH_CLOSE,
    }
    selected_op = st.sidebar.selectbox(
        "Choose Morphological Operation", list(operation_dict.keys()), index=0
    )
    op_code = operation_dict[selected_op]

    shape_dict = {
        "Rectangle": cv2.MORPH_RECT,
        "Cross": cv2.MORPH_CROSS,
        "Ellipse": cv2.MORPH_ELLIPSE,
    }
    selected_shape = st.sidebar.selectbox(
        "Structuring Element Shape", list(shape_dict.keys()), index=0
    )
    shape_code = shape_dict[selected_shape]

    kernel_size = st.sidebar.slider("Kernel Size", min_value=3, max_value=15, value=5, step=2)


    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    else:
        img = create_test_image()
        st.sidebar.info("Using default image with 'ABC' and noise.")

    kernel = cv2.getStructuringElement(shape_code, (kernel_size, kernel_size))

    result = cv2.morphologyEx(img, op_code, kernel)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image (Grayscale)", use_container_width=True)
    with col2:
        st.image(result, caption=f"{selected_op} Result", use_container_width=True)

    st.markdown(f"### Overview of {selected_op}")
    st.write(operation_overview[selected_op])


if __name__ == "__main__":
    main()
