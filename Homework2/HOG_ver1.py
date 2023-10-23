import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_differential_filter():
    # TODO: implement this function
    # 3 by 3 Sobel filter
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return filter_x, filter_y


def filter_image(im, filter):
    # TODO: implement this function
    h, w = im.shape
    k, _ = filter.shape
    padded_im = np.pad(im, k // 2)
    
    im_filtered = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            partial_mat = padded_im[i:i+k, j:j+k]
            partial_mat = np.multiply(partial_mat, filter)
            im_filtered[i][j] = np.sum(partial_mat)

    return im_filtered


def get_gradient(im_dx, im_dy):
    # TODO: implement this function
    im_dx2, im_dy2 = np.square(im_dx), np.square(im_dy)
    grad_mag = np.sqrt(np.add(im_dx2, im_dy2))
    grad_angle = np.arctan2(im_dy, im_dx)
    grad_angle = np.where(grad_angle < 0, grad_angle + np.pi, grad_angle)

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # TODO: implement this function
    h, w = grad_mag.shape
    H, W, D = h // cell_size, w // cell_size, 6
    ori_histo = np.zeros((H, W, D))
    
    bins = [np.deg2rad(d) for d in range(15, 180, 30)]
    ind = np.digitize(grad_angle, bins)
    ind = np.where(ind == 6, 0, ind)
    for i in range((h // cell_size) * cell_size):
        ii = i // cell_size
        for j in range((w // cell_size) * cell_size):
            jj = j // cell_size
            ori_histo[ii][jj][ind[i][j]] += grad_mag[i][j]
    
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # TODO: implement this function
    H, W, D = ori_histo.shape
    HH, WW, DD = H - block_size + 1, W - block_size + 1, D * block_size * block_size
    e = 0.001
    
    ori_histo_normalized = np.zeros((HH, WW, DD))
    for i in range(HH):
        for j in range(WW):
            # concat the original histograms in the block
            for ii in range(block_size):
                for jj in range(block_size):
                    st = D * (ii * block_size + jj)
                    ed = st + D
                    ori_histo_normalized[i, j, st:ed] = ori_histo[i + ii, j + jj, 0:D]
            # normalize
            norm = (np.sum(np.square(ori_histo_normalized[i][j])) + e**2) ** 0.5
            ori_histo_normalized[i][j] /= norm
    
    return ori_histo_normalized


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='red', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    # plt.show()
    plt.savefig('hog.png')


def extract_hog(im, visualize=False, cell_size=8, block_size=2):
    # TODO: implement this function
    filter_x, filter_y = get_differential_filter()
    fim_x, fim_y = filter_image(im, filter_x), filter_image(im, filter_y)
    grad_mag, grad_ang = get_gradient(fim_x, fim_y)
    ori_histo = build_histogram(grad_mag, grad_ang, cell_size)
    hog = get_block_descriptor(ori_histo, block_size)
    
    if visualize:
        visualize_hog(im, hog, cell_size, block_size)
    return hog


def face_recognition(I_target, I_template):
    # TODO: implement this function
    # Use thresholing and NMS(non-maximum suppresion) with IoU 50%
    # bounding boxes: array of (x, y, s)
    # s: NCC(normalized cross-correlation) between the bounding box patch and the template
    threshold_NCC = 0.5
    threshold_IoU = 0.5
    target_h, target_w = I_target.shape
    template_h, template_w = I_template.shape
    template_hog = extract_hog(I_template)
    b = template_hog.flatten()
    b -= np.mean(b)
    
    proposals = []
    # thresholding
    for i in range(target_h - template_h):
        for j in range(target_w - template_w):
            patch = I_target[i:i+template_h, j:j+template_w]
            patch_hog = extract_hog(patch)
            a = patch_hog.flatten()
            a -= np.mean(a)
            s = np.dot(a, b) / (np.linalg.norm(a, 2) * np.linalg.norm(b, 2))
            if s > threshold_NCC:
                proposals.append(np.array([j, i, s]))   # row: [x, y, s]
    # NMS with IoU 50%
    proposals.sort(key=lambda x: x[2], reverse=True)
    bounding_boxes = np.array([[0, 0, 0]])
    while proposals:
        chosen_box = proposals.pop(0)
        bounding_boxes = np.append(bounding_boxes, np.array([chosen_box]), axis=0)

        x1, y1, _ = chosen_box
        x2, y2 = x1 + template_w, y1 + template_h
        new_proposals = []
        for box in proposals:
            # calculate IoU
            x3, y3, _ = box
            x4, y4 = x3 + template_w, y3 + template_h
            x_inter1 = max(x1, x3)
            y_inter1 = max(y1, y3)
            x_inter2 = min(x2, x4)
            y_inter2 = min(y2, y4)
            area_inter = abs(x_inter2 - x_inter1) * abs(y_inter2 - y_inter1)
            area_union = template_h * template_w * 2 - area_inter
            IoU = area_inter / area_union
            if IoU < threshold_IoU:
                new_proposals.append(box)
        proposals = new_proposals
    bounding_boxes = np.delete(bounding_boxes, [0, 0], axis=0)
    
    return bounding_boxes


def visualize_face_detection(I_target, bounding_boxes, box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.imsave('result_face_detection.png', fimg, vmin=0, vmax=1)
    plt.show()


if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im, visualize=False)

    I_target= cv2.imread('target.png', 0) # MxN image

    I_template = cv2.imread('template.png', 0) # mxn  face template

    bounding_boxes = face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png') # MxN image (just for visualization)
    
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0]) # visualization code
