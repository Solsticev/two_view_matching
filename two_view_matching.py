import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist

def read_and_show(img_path1, img_path2, show=False):
    
    image1 = cv2.imread(img_path1)
    image2 = cv2.imread(img_path2)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    print("两张图片的shape分别为：", image1.shape, " ", image2.shape, "\n")
    if show:
        figure, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(image1, cmap='gray')
        ax[1].imshow(image2, cmap='gray')
        plt.show()
        plt.clf()

    return (image1, image2)

def feature_extraction(img1, img2):

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    image_with_keypoints1 = cv2.drawKeypoints(img1, keypoints1, outImage=None)
    image_with_keypoints2 = cv2.drawKeypoints(img2, keypoints2, outImage=None)

    return (keypoints1, descriptors1), (keypoints2, descriptors2)


def brute_force_match(descriptors1, descriptors2, threshold=0.75):
    
    matches = []
    distance_matrix = cdist(descriptors1, descriptors2, "euclidean")
    # min_distance = np.min(distance_matrix, axis=1)
    sorted_distance_index = np.argsort(distance_matrix, axis=1)
    for i in range(len(sorted_distance_index)):
        best_match = distance_matrix[i][sorted_distance_index[i][0]]
        second_best_match = distance_matrix[i][sorted_distance_index[i][1]]
        if best_match < threshold * second_best_match:
            matches.append((i, sorted_distance_index[i][0]))
    print(np.array(matches).shape)

    return matches

def show_matching(matches, img_path1, img_path2, keypoints1, keypoints2, img_name):

    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    hstack_image = np.hstack((img1, img2))

    matched_points1 = np.array([keypoints1[i].pt for i, _ in matches[:200]])
    matched_points2 = np.array([keypoints2[j].pt for _, j in matches[:200]])
    matched_points2 += (img1.shape[1], 0)

    for pt1, pt2 in zip(matched_points1, matched_points2):
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        cv2.line(hstack_image, pt1, pt2, (128, 0, 0))

    plt.imshow(cv2.cvtColor(hstack_image, cv2.COLOR_BGR2RGB))
    
    plt.savefig("./res/" + img_name + "_matching.png", dpi=300)
    plt.clf()
    # plt.show()


def cal_fundamental_mtx(pts1, pts2, method):
    if method == "FM_LMEDS":
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    elif method == "FM_8POINT":
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT) 

    elif method == "8POINT":

        # 自己实现的normalized 8 POINT
        mean_pts1 = np.mean(pts1, axis=0)
        mean_pts2 = np.mean(pts2, axis=0)
        scale_pts1 = pts1 - mean_pts1
        scale_pts2 = pts2 - mean_pts2

        s_1 = 0
        s_2 = 0
        length = len(pts1)
        for i in range(length):
            s_1 += np.sqrt(scale_pts1[i,0]**2 + scale_pts1[i,1]**2)
            s_2 += np.sqrt(scale_pts1[i,0]**2 + scale_pts2[i,1]**2)
        
        s_1 = 1 / (len(pts1)*np.sqrt(2)) * s_1
        s_2 = 1 / (len(pts1)*np.sqrt(2)) * s_2

        for i in range(length):
            scale_pts1[i,:] = 1/s_1*scale_pts1[i,:]
            scale_pts2[i,:] = 1/s_1*scale_pts2[i,:]
        
        # T_1和T_1用于规范化F
        T_1 = np.array([
            [1/s_1, 0, 1/s_1*(-mean_pts1[0])],
            [0, 1/s_1, 1/s_1*(-mean_pts1[1])],
            [0,0,1]
        ])
        T_2 = np.array([
            [1/s_2, 0, 1/s_2*(-mean_pts2[0])],
            [0, 1/s_2, 1/s_2*(-mean_pts2[1])],
            [0,0,1]
        ])

        A = []
        for i in range(length):
            x1, y1 = scale_pts1[i]
            x2, y2 = scale_pts2[i]
            # A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
            A.append([x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1])
        A = np.array(A)

        U, S, Vt = np.linalg.svd(A)
        # F是A^TA中最小特征值对应的元素，Vt的行向量是A^TA的特征向量
        F = Vt[-1].reshape(3, 3)
        U, S, Vt = np.linalg.svd(F)
        # 使F的秩为2，将最小奇异值设置为0
        S[-1] = 0
        F = U @ np.diag(S) @ Vt
        # 规范化F
        F = np.transpose(T_2) @ F @ T_1
        # 标准化基础矩阵
        # F /= F[2, 2]
        mask = "nil"
    else:
        print("invalid method!")
        assert(0)

    print("基础矩阵为", F, "\n")
    return F, mask

# 计算极线
def compute_epilines(F, points):

        # 构造齐次坐标，将每个点扩展为 [x, y, 1]
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

        # 计算极线
        epilines = np.dot(F, points_homogeneous.T)

        # 规范化极线
        norm = np.sqrt(epilines[0] ** 2 + epilines[1] ** 2)
        epilines /= norm

        return epilines.T

def show_epipolar_lines(img1, img2, lines, pts1, pts2): 
    
    r, c = img1.shape 
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) 
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) 
      
    for r, pt1, pt2 in zip(lines, pts1, pts2): 
          
        color = tuple(np.random.randint(0, 255, 3).tolist()) 
          
        x0, y0 = map(int, [0, -r[2] / r[1] ]) 
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1] ]) 
          
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1) 
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1) 
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1) 
    return img1, img2 

    
if __name__ == "__main__":

    img_name = "building"
    img_path1 = "./pics/" + img_name + "1.jpg"
    img_path2 = "./pics/" + img_name + "2.jpg"
    method = "FM_LMEDS" # 计算fundamental matrix的方法
    (image1, image2) = read_and_show(img_path1, img_path2, show=False)

    print("开始进行特征提取\n")
    (keypoints1, descriptors1), (keypoints2, descriptors2) = feature_extraction(image1, image2)
    print("特征提取完成\n")
    
    print("开始进行特征匹配\n")
    matches = brute_force_match(descriptors1, descriptors2, threshold=0.75)
    print("特征匹配完成\n")
    
    show_matching(matches, img_path1, img_path2, keypoints1, keypoints2, img_name=img_name)

    matched_pts1 = np.int32([keypoints1[i].pt for i, _ in matches])
    matched_pts2 = np.int32([keypoints2[j].pt for _, j in matches])

    # 根据匹配结果计算fundamental matrix
    print("开始进行fundamental matrix和epipolar lines的计算\n")
    F, mask = cal_fundamental_mtx(matched_pts1, matched_pts2, method)

    if method == "FM_LMEDS" or method == "FM_8POINT":
        matched_pts1 = matched_pts1[mask.ravel() == 1] 
        matched_pts2 = matched_pts2[mask.ravel() == 1]
        pass

    # linesLeft = cv2.computeCorrespondEpilines(matched_pts2.reshape(-1, 1, 2), 2, F) 
    # linesRight = cv2.computeCorrespondEpilines(matched_pts1.reshape(-1, 1, 2), 1, F)

    # 计算第一张图像中的特征点的极线
    linesRight = compute_epilines(F, matched_pts1)

    # 计算第二张图像中的特征点的极线
    linesLeft = compute_epilines(F.T, matched_pts2)

    linesRight = linesRight.reshape(-1, 3)
    linesLeft = linesLeft.reshape(-1, 3) 
     
    img5, img6 = show_epipolar_lines(image1, image2, linesLeft, matched_pts1, matched_pts2) 
    img3, img4 = show_epipolar_lines(image2, image1, linesRight, matched_pts2, matched_pts1) 
    
    plt.subplot(221), plt.imshow(img5) 
    plt.subplot(222), plt.imshow(img6) 
    plt.subplot(223), plt.imshow(img3) 
    plt.subplot(224), plt.imshow(img4) 
    plt.savefig("./res/" + img_name + "_epipolar_lines_" + method + ".png", dpi=300)
    # plt.show() 
    print("计算完毕，结果保存在res目录下")