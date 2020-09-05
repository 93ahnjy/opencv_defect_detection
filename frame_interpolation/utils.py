

def sub_block(input_img, start, end):
    if start[0] > end[0] or start[1] > end[1]:
        raise ValueError('start: {}, end: {}'.format(start, end))
    return input_img[start[0]:end[0], start[1]:end[1]]

def validate_block(block_idx, block_jdx, vy, vx, height, width, N):
    start_current_block = (block_idx + vy, block_jdx + vx)
    end_current_block = (block_idx + vy + N, block_jdx + vx + N)
    start_previous_block = (block_idx - vy, block_jdx - vx)
    end_previous_block = (block_idx - vy + N, block_jdx - vx + N)

    if start_current_block[0] < 0 or start_current_block[1] < 0 or start_previous_block[0] < 0 or start_previous_block[1] < 0:
        return False, None, None, None, None
    if end_current_block[0] > height or end_current_block[1] > width or end_previous_block[0] > height or end_previous_block[1] > width:
        return False, None, None, None, None

    return True, start_current_block, end_current_block, start_previous_block, end_previous_block


if __name__ == "__main__":
    import numpy as np
    input_img = np.zeros([256, 256, 3], dtype=np.float32)
    sub_block(input_img, (3, 3), (2, 5))
    pass