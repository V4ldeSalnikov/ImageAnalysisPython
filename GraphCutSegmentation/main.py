import streamlit as st
import numpy as np
import scipy
from scipy.sparse.csgraph import maximum_flow
from scipy import sparse
import matplotlib.pyplot as plt


def edges4connected(height, width, only_one_dir=0):
    #
    # Generates a 4-connectivity structure for a height*width grid
    #
    # if only_one_dir==1, then there will only be one edge between node i and
    # node j. Otherwise, both i-->j and i<--j will be added.
    #
    N = height * width
    I = np.array([])
    J = np.array([])

    # connect vertically (down, then up)
    iis = np.delete(np.arange(N), np.arange(height - 1, N, height))
    jjs = iis + 1
    if ~only_one_dir:
        I = np.hstack((I, iis, jjs))
        J = np.hstack((J, jjs, iis))
    else:
        I = np.hstack((I, iis))
        J = np.hstack((J, jjs))

    # connect horizontally (right, then left)
    iis = np.arange(0, N - height)
    jjs = iis + height
    if ~only_one_dir:
        I = np.hstack((I, iis, jjs))
        J = np.hstack((J, jjs, iis))
    else:
        I = np.hstack((I, iis))
        J = np.hstack((J, jjs))

    return I, J



@st.cache_data
def load_data():
    data = scipy.io.loadmat('heart_data.mat')
    data_chamber = data['chamber_values'].flatten()
    data_background = data['background_values'].flatten()
    im = data['im']
    return im, data_chamber, data_background



def graph_cut_segmentation(im, data_chamber, data_background, v):

    # --- Calculate means and stds ---
    m_chamber = np.mean(data_chamber)
    s_chamber = np.std(data_chamber)
    m_background = np.mean(data_background)
    s_background = np.std(data_background)

    M, N = im.shape

    # --- Setting up edge structures and weight on edges ---
    Ie, Je = edges4connected(M, N)  # 4-connected graph
    Ve = v * np.ones_like(Ie)       # Regularization term weight

    # --- Negative log-likelihoods for chamber and background ---
    Vs = ((im.flatten() - m_chamber) ** 2) / (2 * s_chamber ** 2)
    Vt = ((im.flatten() - m_background) ** 2) / (2 * s_background ** 2)

    # --- Source & sink connections ---
    Is1, Js1 = np.arange(M * N), (M * N) * np.ones(M * N)
    Is2, Js2 = (M * N) * np.ones(M * N), np.arange(M * N)

    It1, Jt1 = np.arange(M * N), (M * N + 1) * np.ones(M * N)
    It2, Jt2 = (M * N + 1) * np.ones(M * N), np.arange(M * N)

    # Combine edges
    I = np.hstack((Ie, Is1, Is2, It1, It2)).astype(np.int32)
    J = np.hstack((Je, Js1, Js2, Jt1, Jt2)).astype(np.int32)
    V = np.hstack((Ve, Vs, Vs, Vt, Vt))

    # ---: Build sparse matrix ---
    sf = 5000
    V = np.round(V * sf).astype(np.int32)
    F = sparse.coo_array((V, (I, J)), shape=(M * N + 2, M * N + 2)).tocsr()

    # ---  Solve max-flow/min-cut ---
    mf = maximum_flow(F, M * N, M * N + 1)

    # ---  Extract & visualize segmentation ---
    seg = mf.flow
    # seg[:M*N, M*N+1] is the flow from each pixel node to the sink
    imflow = seg[:M * N, M * N + 1].reshape((M, N)).toarray().astype(float)
    # Compare that flow to the sink capacity from Step 4:
    imseg = imflow < V[-M * N:].reshape(M, N)

    return imseg


# ----------------------------------------------------------------
# Streamlit main function
# ----------------------------------------------------------------
def main():
    st.title("Graph-Cut Heart Chamber Segmentation")
    st.write("""
    This demo uses a 4-connected graph-cut segmentation on heart MRI data.
    Adjust the regularization parameter \\(v\\) in the sidebar to see how
    the segmentation changes.
    """)

    # Load data once
    im, data_chamber, data_background = load_data()

    # Let user select regularization parameter v
    v = st.sidebar.slider("Regularization parameter (v)", min_value=1, max_value=30, value=7, step=1)

    # Run graph-cut segmentation
    imseg = graph_cut_segmentation(im, data_chamber, data_background, v)

    # Show results side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Original image
    axs[0].imshow(im, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Segmentation result
    axs[1].imshow(imseg, cmap='gray')
    axs[1].set_title(f"Segmentation (v={v})")
    axs[1].axis('off')

    st.pyplot(fig)

    algorithm_description = r"""
        **Algorithm Steps:**

        1. **Calculate Means and Standard Deviations**  
           Compute the mean $(\mu)$ and standard deviation $(\sigma)$ of the pixel intensities
           for both the heart chamber and the background.

        2. **Setting Up Edge Structures and Weight on Edges**  
           Build a 4-connected grid where adjacent pixels are linked by edges. Assign a regularization
           weight $v$ to these edges, to increase spatial smoothness.

        3. **Negative Log Likelihoods**  
           For each pixel $p$, compute the negative log-likelihoods:
           $$
           V_s(p) = \frac{(I(p) - \mu_{\text{chamber}})^2}{2\sigma_{\text{chamber}}^2}, \quad
           V_t(p) = \frac{(I(p) - \mu_{\text{background}})^2}{2\sigma_{\text{background}}^2}
           $$
           These represent the cost of assigning \(p\) to chamber or background.

        4. **Source and Sink Connections**  
           Connect each pixel node to a source node with capacity $V_s(p)$ and to a sink node with capacity
           $V_t(p)$. This encodes how likely each pixel is to belong to chamber vs. background.

        5. **Building the Sparse Matrix**  
           Combine all edges (pixel-to-pixel, pixel-to-source, pixel-to-sink) into a sparse adjacency matrix
           for the flow network.

        6. **Max-flow/Min-cut**  
           Running the max-flow algorithm to partition the graph into two sets:
           those connected to the source (chamber) and those connected to the sink (background).

        7. **Extract and Visualize Segmentation**  
           From the flow network, determine which pixels remain in the source set. This creates a binary
           segmentation mask for the heart chamber.
        """
    st.markdown(algorithm_description)

if __name__ == "__main__":
    main()
