import numpy as np
from PIL import Image

def calculate_transition_matrix(sequence:list):
    '''
    Calculate the state transition frequency matrix (STFM) of a given binary byte sequence.

    Args:
        sequence (list): Binary byte sequence.

    Returns:
        list: Two-dimensional STFM.
    '''
    n_states = 256
    transition_counts = np.zeros((n_states, n_states), dtype=int)
    
    for i in range(len(sequence) - 1):
        from_state = sequence[i]
        to_state = sequence[i + 1]
        from_index = from_state
        to_index = to_state
        transition_counts[from_index][to_index] += 1

    transition_matrix = transition_counts

    return transition_matrix

def binary_to_img(file_path, image_path, t1=255, t2=255):
    """
    The given binary file is mapped to an RGB image by calculating the state transition frequency matrix (STFM) and applying the DT-LHTA algorithm.

    Args:
        file_path (str): Given the binary file path.
        image_path (str): RGB image output path.
        t1 (int): B-channel threshold.
        t2 (int): G-channel threshold.
    Returns:
        None.
    """

    sequence = []
    with open(file_path, mode='rb') as bytes:
        for bytes_line in bytes.readlines():
            sequence += [byte for byte in bytes_line]

    matrix = calculate_transition_matrix(sequence)

    rows, cols = len(matrix), len(matrix[0])
    image = np.zeros((3,256,256))
    for i in range(rows):
        for j in range(cols):
            element = matrix[i][j]
            if element > t1:
                image[0][i][j] = t1
                element = element - t1
                if element > t2:
                    image[1][i][j] = t2
                    image[2][i][j] = np.log2(element - t2)
                else:
                    image[1][i][j] = element
            else:
                image[0][i][j] = element

    image[2] = np.clip(image[2], 0, 255)
     
    b = Image.fromarray(np.uint8(image[0]))
    g = Image.fromarray(np.uint8(image[1]))
    r = Image.fromarray(np.uint8(image[2]))
    Image.merge("RGB", (r, g, b)).save(image_path)


if __name__  == '__main__':
    # binary_to_img(file_path, image_path)
    print('over.')
