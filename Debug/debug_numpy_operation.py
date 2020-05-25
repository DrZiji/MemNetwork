import numpy as np

def bbox_ori2fea(bbox, cog_field, stride):
    """
    :param bbox: ndarray float [batch, 4].在图片中的目标框
    :param cog_field: int 当前特征图中一个点代表的感受野
    :param stride: 获得当前特征网络的stride
    :return ndarray int [batch, 4].在特征中的目标框
    """
    left_up_x = bbox[:, 0][:, np.newaxis]
    left_up_y = bbox[:, 1][:, np.newaxis]
    right_down_x = (bbox[:, 0] + bbox[:, 2])[:, np.newaxis]
    right_down_y = (bbox[:, 1] + bbox[:, 3])[:, np.newaxis]

    temp = np.concatenate((left_up_x, left_up_y, right_down_x, right_down_y), axis=1)
    temp = (temp - cog_field) / stride + 1
    temp = temp.astype(np.int)
    bbox_fea = np.where(temp > 0, temp, 0)
    return bbox_fea

if __name__ == "__main__":
    # A = np.array([[5, 5, 20, 20],
    #               [8, 3, 30, 29]])
    # A_fea = bbox_ori2fea(A, 10, 2)
    t = range(10)
    print()
