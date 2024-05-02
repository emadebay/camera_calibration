import numpy as np
import cv2

# world_coordinates = {
#     1 : (12,0,2),
#     2 :(12,0,0),
#     3 : (8,0,0),
#     4 : (4,0,2),
#     5 : (0,4,2),
#     6 : (0,4,0)
# }


# print(world_coordinates)

image_path = 'calibration-rig.jpg'
image = cv2.imread( image_path)
cv2.imshow('Image', image)

world_coordinates = [(18,0,0), (18,0,12), (0,16,0), (0,16,14), (0,4,14), (4,0,6)]
image_coordinates = []
b_matrix = np.zeros((12,12))

#put image in numpy
image_np = np.array(image)
def pick_image_coordinates(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        image_height = image_np.shape[0]
        print(image_np.shape)
        #x and y from open cv happens to be row and column
        #x = column
        #y = image height minus row

        x_cartesian = x
        y_cartesian = image_height - y

        # x_cartesian = y
        # y_cartesian = image_height - x

        image_coordinates.append((x_cartesian,y_cartesian))
        # image_coordinates.append((x,y))

        print((x,y), (x_cartesian,y_cartesian))


cv2.setMouseCallback('Image', pick_image_coordinates)



cv2.waitKey(0)
cv2.destroyAllWindows()

# print("world coord", world_coordinates)
# print("img coord",image_coordinates)





def setup_the_12_by_12_b_matrix():
    
    #each data point takes up to rows in the b matrix
    #i used recp to denote the indexing system
    recp = 0

    for i in range(len(world_coordinates)):
        world_coordinates_i = world_coordinates[i]
        x_w = world_coordinates_i[0]
        y_w = world_coordinates_i[1]
        z_w = world_coordinates_i[2]

        image_coordinates_i = image_coordinates[i]
        x_i = image_coordinates_i[0]
        y_i = image_coordinates_i[1]
        
        # first row of each points
        b_matrix[recp,0] = x_w
        b_matrix[recp,1] = y_w
        b_matrix[recp,2] = z_w
        b_matrix[recp,3] = 1
        b_matrix[recp,4] = 0
        b_matrix[recp,5] = 0
        b_matrix[recp,6] = 0
        b_matrix[recp,7] = 0
        b_matrix[recp,8] = (-x_i)*x_w
        b_matrix[recp,9] = (-x_i)*y_w
        b_matrix[recp,10] = (-x_i)*z_w
        b_matrix[recp,11] = -x_i

        #second row of each points
        #increment the counting system by one
        recp = recp + 1
        b_matrix[recp,0] = 0
        b_matrix[recp,1] = 0
        b_matrix[recp,2] = 0
        b_matrix[recp,3] = 0
        b_matrix[recp,4] = x_w
        b_matrix[recp,5] = y_w
        b_matrix[recp,6] = z_w
        b_matrix[recp,7] = 1
        b_matrix[recp,8] = (-y_i)*x_w
        b_matrix[recp,9] = (-y_i) * y_w
        b_matrix[recp,10] = (-y_i) * z_w
        b_matrix[recp,11] =  -y_i 

        recp = recp + 1


setup_the_12_by_12_b_matrix()


def calculate_the_projection_matrix():

    #solving using the aproach 2 as discussed in class
    b_matrix_multiplied_by_b_matrix_transpose = np.matmul(b_matrix, b_matrix.T)

    #compute the smallest eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(b_matrix_multiplied_by_b_matrix_transpose)

    #find the inde of the smallest eigenvalue
    index_min_eigenvalue = np.argmin(eigenvalues)

    #get the corresponding eigenvector
    eigenvector_min = eigenvectors[:, index_min_eigenvalue]

    #get it into its corresponding 3 by  4
    reshaped_matrix = np.reshape(eigenvector_min, (3,4))

    #calculate the euclidean norm of r31, r32, r33
    r31, r32, r33 = reshaped_matrix[2,0], reshaped_matrix[2,1], reshaped_matrix[2,2]
    #euclid norm of the first 3 elements of the last row
    #denoted as enof
    enof = np.linalg.norm([r31, r32, r33])

    #get The final projection matrix
    projection_matrix = reshaped_matrix / enof

    # print(enof)
    # print(reshaped_matrix)
    # print(eigenvector_min)
    # print(projection_matrix)

    return projection_matrix

def recover_the_intrinsics(intrinsic_parameters_matrix):
    # Center of projection u_0, v_0
    u_0 = intrinsic_parameters_matrix[0, 2]
    v_0 = intrinsic_parameters_matrix[1, 2]

    # Focal length multiplied by scale factor
    # beta_sv_f = np.sqrt(intrinsic_parameters_matrix[1, 1] - (intrinsic_parameters_matrix[1, 2] ** 2))
    # alpha_su_f = np.sqrt(intrinsic_parameters_matrix[0, 0] - (intrinsic_parameters_matrix[0, 2] ** 2))

    beta_sv_f = np.sqrt(intrinsic_parameters_matrix[1, 1] - (intrinsic_parameters_matrix[1, 2] ** 2))
    alpha_su_f = np.sqrt(intrinsic_parameters_matrix[0, 0] - (intrinsic_parameters_matrix[0, 2] ** 2))

    # Assuming skew is zero
    skew = 0

    # Construct the 3x3 matrix with parameters
    K = np.array([[alpha_su_f, skew, u_0],
                  [0, beta_sv_f, v_0],
                  [0, 0, 1]])

    print("intrinsic before decoupling")
    print(intrinsic_parameters_matrix)

    return K

def recover_the_intrinsic_and_extrinsic_parameter(projection_matrix):
    #The projection matrix of the form
    #[B b] where B is a 3 by 3 matrix
    #k is a 3 by 1 column vector

    #Inside the b matrix is the intrinsic K and the rotation R
    #that is B = KR
    B = projection_matrix[:, :3]
    #inside the b column vector is the intrinsic K and 
    #translation vector t
    #that is, b = Kt
    b = projection_matrix[:, -1]

    #using QR factorization Q is the intrinsic and R is the rotation matrix
    Q_rotation, R_K_intrinsic = np.linalg.qr(B)
    translation_QR = np.matmul(np.linalg.inv(R_K_intrinsic), b)

    print("Q_rotation", Q_rotation)
    print("\n")
    print("R_K_intrinsic", R_K_intrinsic)
    

    #the intrinsic parameters matrix is of 
    #of the form A = B(B)^T
    #because R is orthornaml, that Rr^T = I
    A = np.matmul(B, B.T)

    #using Q and R factorization
    # K, R = np.linalg.qr(B)

    #The projection matrix is scaled up to the last element of A
    #that is A33
    scale_factor = A[2,2]
    #normalize the intrinsic matrix by the scale
    intrinsic_matrix_jumbled_up = A / scale_factor

    #recover the intrinsic matrix k assuming no skew
    #that is s is zero
    K = recover_the_intrinsics(intrinsic_matrix_jumbled_up)

    #get the rotation matrix
    rotation_matrix = np.matmul(np.linalg.inv(K), B)
    #get the translation vector
    translation = np.matmul(np.linalg.inv(K), b)

    # print("projection_matrix")
    # print("\n")
    # print(projection_matrix)
    # print("\n")

    #add the translation matrix
    #as part of the column of the rotation matrix
    column_added = np.hstack((rotation_matrix, translation.reshape(-1,1)))
    #multiplied the intrinsic by the extrinsic
    #it should be the same as the projection matrix

    verification = np.matmul(B, column_added)
    # print("verification")
    # print("\n")
    # print(verification)
    

    #return
    #project matrix - not decoupled
    #intrinsic matrix - has the alpha (focal length), beta(focal length), center of projection(u,v) and skew(distortion)
    #The rotation matrix
    #the tranlsation vector tx,ty,tz
    return projection_matrix, K, rotation_matrix, translation
    # return projection_matrix, R_K_intrinsic, Q_rotation, translation_QR

def reproject_the_world_coordinates(K_intrinsic, rotation_extrinsic, tranlsation_extrinsic, world_coordinates):
    #make sure the translation vector is in the shape of 3 by 1
    tranlsation_extrinsic = tranlsation_extrinsic.reshape(-1,1)
    #combined the whole extrinsic parameters by adding the translation
    #vector to the end of rotation matrix. resulting in 3 by 4
    whole_extrinsic = np.hstack((rotation_extrinsic, tranlsation_extrinsic))
    #multiplied intrinsic and extrinic
    intrinsic_and_extrinsic = np.matmul(K_intrinsic, whole_extrinsic)

    #holds the reprojected points
    reproj_points = []
    for point in world_coordinates:
        x, y, z = point
        column_vector_of_points = np.array([[x], [y], [z], [1]])
        x_y_reprojection = np.matmul(intrinsic_and_extrinsic, column_vector_of_points)

        #we get an homogenous cooridnates such
        #that the last number is a scale factor
        scalar = x_y_reprojection[2,0]
        #normalize
        scaled_x_y = x_y_reprojection / scalar
        x = scaled_x_y[0,0]
        y = scaled_x_y[1,0]
        p = (x,y)
        reproj_points.append(p)
    return reproj_points


projection_matrix = calculate_the_projection_matrix()
proj_matrix,  K, rot_matrix, tx_tx_ty = recover_the_intrinsic_and_extrinsic_parameter(projection_matrix)

reprojected_points = reproject_the_world_coordinates(K, rot_matrix, tx_tx_ty, world_coordinates)

print("The list of x, y coordinates of the image points  picked")
print(image_coordinates)
print("\n")

print("The list of x, y coordinates of the reprojections points")
print(reprojected_points)
print("\n")

print("\n")
print("the projection matrix")
print(proj_matrix)

print("\n")
print("K intrinsic matrix")
print(K)

print("\n")
print("the rotation matrix: extrinsic parameters")
print(rot_matrix)

print("\n")
print("the translation vector: extrinsic parameters")
print(tx_tx_ty)
print("\n")



