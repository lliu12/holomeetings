# utility functions for hARvARd ES143 project

# Import any required libraries here
import cv2                               # OpenCV
import numpy as np                       # numpy
import os
from pupil_apriltags import Detector
import requests
import pickle
import glob   # filename and path management for file I/O
# for 3D plotting (better than matplotlib's mplot3d)
import plotly.graph_objects as go
from matplotlib import animation
from IPython.display import HTML
import io
from PIL import Image, ImageDraw  # use pip install pillow to install this
from tqdm import tqdm

import matplotlib.pyplot as plt
# Modify this line to adjust the displayed plot size. You can also call
# it with different parameters or use plt.figure(figsize=[H, W]) before specific plots.
plt.rcParams['figure.figsize'] = [10, 10]

data = pickle.load(open('./AprilBoards.pickle', 'rb'))
at_coarseboard = data['at_coarseboard']
at_fineboard = data['at_fineboard']
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# CONSTANTS:
NUM_REFERENCE_FRAMES = 10


def add_camera(h, w, camera, raysize, figobj):
    # Add tetrahedral camera to pyplot figure
    # h,w:      height and width of image in pixels
    # camera:   3x4 camera matrix
    # raysize:  length of tetrahedral edges (in world units)
    # fig:      pyplot figure object
    #
    # Returns: 1
    #
    # Uses anatomy of camera matrices from Hartley and Zisserman Chapter 6

    # normalize camera such that bottom-left three-vector
    #   corresponds to unit-length principal ray in front of camera (HZ Section 6.2.3)
    camera = camera * \
        np.sign(np.linalg.det(camera[:, 0:3]))/np.linalg.norm(camera[2, 0:3])

    # Compute camera center (null vector of P)
    _, _, v = np.linalg.svd(camera)
    C = np.transpose(v[-1, 0:3]) / v[-1, 3]

    # Back-project image corners to unit-length 3D ray segments:
    S = np.array([[0, 0, 1],       # homog image coords if top left pixel
                  [0, h-1, 1],     # bottom left
                  [w-1, h-1, 1],   # bottom right
                  [w, 0, 1]])      # top right

    #   HZ equation (6.14): compute one 3D point along each ray
    X = np.transpose(np.linalg.lstsq(
        camera[:, 0:3],
        np.transpose(S)-np.expand_dims(camera[:, 3], axis=1),
        rcond=None)[0])

    #   unit-vectors from camera center to each 3D point
    V = X - np.tile(C, (4, 1))
    V = V / np.linalg.norm(V, ord=2, axis=1, keepdims=True)

    # make sure these vectors point forwards from the camera instead of backwards
    V = V*np.expand_dims(np.sign(np.sum(V *
                                        np.tile(camera[2, 0:3], (4, 1)), axis=1)), axis=1)

    #   desired ray segments that are length raysize in these directions
    V = np.tile(C, (4, 1)) + raysize * V

    # append the camera center itself to complete the four tetrahedral vertices
    V = np.vstack([C, V])

    # add camera center to figure
    figobj.add_trace(go.Scatter3d(
        x=[C[0]],
        y=[C[1]],
        z=[C[2]],
        mode='markers',
        marker=dict(
            size=3,
            color='#ff7f0e'
        )
    )
    )

    # add tetrahedron to figure
    figobj.add_trace(go.Mesh3d(
        # vertices of tetrahedron
        x=V[:, 0],
        y=V[:, 1],
        z=V[:, 2],

        # i, j and k give the vertices of triangles
        i=[0, 0, 0, 0],
        j=[1, 2, 3, 4],
        k=[2, 3, 4, 1],
        opacity=0.5,
        color='#ff7f0e'
    ))

    return 1

# RANSAC PLANE FIT
# input: list of x, y, and z coordinates of data points to fit a best plane to
# output:
# best_plane: homogenous a,b,c,d defining plane
# best_consensus_set: list of indices of points on plane, so X[best_consensus_set] would give the consensus points


def plane_fit_ransac(x, y, z, iterations=400, INLIER_THRESH=.1):
    best_consensus_set = np.zeros((0,))
    numpts = x.size
    best_plane = None

    # try a bunch of different planes
    for _ in range(iterations):
        # Randomly sample three distinct points
        sample_inds = np.random.choice(numpts, size=3, replace=False)
        xs = x[sample_inds]
        ys = y[sample_inds]
        zs = z[sample_inds]

        # Make homogeneous
        p1 = np.array([xs[0], ys[0], zs[0], 1], dtype=np.float32)
        p2 = np.array([xs[1], ys[1], zs[1], 1], dtype=np.float32)
        p3 = np.array([xs[2], ys[2], zs[2], 1], dtype=np.float32)

        # Get plane determined by these three points
        A = np.vstack([p1, p2, p3])
        u, s, vt = np.linalg.svd(A)
        plane = vt[-1, :]

        # store perpendicular distance from plane to every pt in dist array
        dist = np.abs(x * plane[0] + y * plane[1] + z * plane[2] + plane[3]
                      ) / np.sqrt(plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)

        # Find consensus set (tuple of indices within x (and y))
        consensus_set = np.nonzero(dist < INLIER_THRESH)[0]

        # If it is larger than the best one so far, keep it
        if consensus_set.size >= best_consensus_set.size:
            best_consensus_set = consensus_set
            best_plane = plane

    # inliers for best line
    xc = x[best_consensus_set]
    yc = y[best_consensus_set]
    zc = z[best_consensus_set]

    # fit final line to these inliers
    # ok lol this better plane fitting part isn't working but that's fine
    p = plane_fit(np.vstack((xc, yc, zc)).T)

    return best_plane, best_consensus_set

# combine channels into one color image


def RGB2Color(imgR, imgG, imgB):
    return np.stack((imgR, imgG, imgB), axis=-1)

# Convert from Nxm inhomogeneous to Nx(m+1) homogeneous coordinates


def in2hom(X):
    return np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)

# Convert from Nxm homogeneous to Nx(m-1) inhomogeneous coordinates


def hom2in(X):
    return X[:, :-1] / (X[:, -1:] + 0.000000001)


def detect_aprilboard(img, board, apriltag_detector):
    # Usage:  imgpoints, objpoints, tag_ids = detect_aprilboard(img,board,AT_detector)
    #
    # Input:
    #   image -- grayscale image
    #   board -- at_coarseboard or at_fineboard (list of dictionaries)
    #   AT_detector -- AprilTag Detector parameters
    #
    # Returns:
    #   imgpoints -- Nx2 numpy array of (x,y) image coords
    #   objpoints -- Nx3 numpy array of (X,Y,Z=0) board coordinates (in inches)
    #   tag_ids -- Nx1 list of tag IDs

    imgpoints = []
    objpoints = []
    tagIDs = []

    # detect april tags
    tags = apriltag_detector.detect(img,
                                    estimate_tag_pose=False,
                                    camera_params=None,
                                    tag_size=None)

    if len(tags):
        # collect image coordinates of tag centers
        imgpoints = np.vstack([sub.center for sub in tags])

        # list of all detected tag_id's in image
        tagIDs = [sub.tag_id for sub in tags]

        # all board list-elements that contain one of the detected tag_ids
        objs = list(filter(lambda tagnum: tagnum['tag_id'] in tagIDs, board))

        if len(objs):
            # gather the center coordinates from each of these board elements
            objpoints = np.vstack([sub['center'] for sub in objs])

    return imgpoints, objpoints, tagIDs


# get plane of aprilboard in an image, given camera cal info
def computePlane(calMatrix, distCoeffs, img, board, debug=False):

    at_detector = Detector(families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

    # Fine
    imgpoints_fine, objpoints_fine, tagids_fine = detect_aprilboard(
        img, board, at_detector)
    if debug:
        print("fine: {} imgpts, {} objpts".format(
            len(imgpoints_fine), len(objpoints_fine)))

    orig = img
    if len(orig.shape) == 3:
        img = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    else:
        img = orig

    # compute normalized image coordinates (equivalent to K^{-1}*x)
    imgpts_fine_norm = cv2.undistortPoints(
        imgpoints_fine, calMatrix, distCoeffs)

    # homographies from each board to normalized image pts
    H_fine, _ = cv2.findHomography(objpoints_fine[:, :2], imgpts_fine_norm)

    # extract rotation and translation from homography: fineboard
    H_fine = 2*H_fine / \
        (np.linalg.norm(H_fine[:, 0]) + np.linalg.norm(H_fine[:, 1]))
    R_fine = np.hstack((H_fine[:, :2], np.atleast_2d(
        np.cross(H_fine[:, 0], H_fine[:, 1])).T))
    rvec_fine, _ = cv2.Rodrigues(R_fine)
    R_fine, _ = cv2.Rodrigues(rvec_fine)
    tvec_fine = H_fine[:, 2:3]

    # get plane equation which is [r3, -r3.T@T]
    r3_fine = R_fine[:, 2:3]
    prod_fine = -r3_fine.T @ tvec_fine
    plane_fine = np.concatenate((r3_fine, prod_fine), axis=0)

    return plane_fine

# Calibrates the camera, gets camera matrices, etc.


def calibrateCamera(video, board, debug=False):

    # initialize 3D object points and 2D image points
    calObjPoints = []
    calImgPoints = []

    # set up april tag detector (I use default parameters; seems to be OK)
    at_detector = Detector(families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

    # pick out 10 frames for camera calibration
    frames = np.random.choice(
        video.shape[2], NUM_REFERENCE_FRAMES, replace=False)

    if debug:
        # create figure to plot detected tags
        fig, axs = plt.subplots(
            np.ceil(NUM_REFERENCE_FRAMES/2).astype(int), 2, figsize=(15, 50))

    for count, ind in enumerate(frames):

        img = video[:, :, ind]

        if debug:
            # show image
            axs.reshape(-1)[count].imshow(img / 255.0, cmap="gray")
            axs.reshape(-1)[count].set_axis_off()
            axs.reshape(-1)[count].set_title("Image {}".format(count))

        # detect apriltags and report number of detections
        imgpoints, objpoints, tagIDs = detect_aprilboard(
            img, board, at_detector)
        if debug:
            print("{} {}: {} imgpts, {} objpts".format(
                count, ind, len(imgpoints), len(objpoints)))

        # append detections if some are found
        if len(imgpoints) and len(objpoints):

            if debug:
                # display detected tag centers
                axs.reshape(-1)[count].scatter(imgpoints[:, 0],
                                               imgpoints[:, 1], marker='o', color='#ff7f0e')

            # append points detected in all images, (there is only one image now)
            calObjPoints.append(objpoints.astype('float32'))
            calImgPoints.append(imgpoints.astype('float32'))

    if debug:
        plt.show()

    # convert to numpy array
    calObjPoints = np.array(calObjPoints)
    calImgPoints = np.array(calImgPoints)

    # calibrate the camera
    reprojerr, calMatrix, distCoeffs, calRotations, calTranslations = cv2.calibrateCamera(
        calObjPoints,
        calImgPoints,
        img.shape,    # image H,W for initialization of the principal point
        None,         # no initial guess for the remaining entries of calMatrix
        None,         # initial guesses for distortion coefficients are all 0
        flags=None)  # default contstraints (see documentation)

    if debug:
        # Print reprojection error. calibrateCamera returns the root mean square (RMS) re-projection error in pixels.
        # Bad calibration if this value if too big
        print('RMSE of reprojected points:', reprojerr)
        print('Distortion coefficients:', distCoeffs)
        print('Intrinsic camera matrix', calMatrix)

    return (reprojerr, calMatrix, distCoeffs, calRotations, calTranslations)

# Gets virtual object from an OBJ file


def getObject(src_path):
    file = open(src_path)

    vertices = []
    faces = []

    # parse .obj text to get vertices and faces
    for l in (file.read().splitlines()):
        line = l.split()
        if len(line) > 0 and line[0] == 'v':
            vertices.append(line[1:])
        elif len(line) > 0 and line[0] == 'f':
            face = []
            for elt in line[1:]:
                i = elt.index('/')
                face.append(int(elt[:i]))
#             if len(face) == 4:
#                 faces.append(np.array(face))
            if len(face) == 3:
                faces.append(np.array(face))
            elif len(face) == 4:
                faces.append(np.array(face[0:3]))
                faces.append(np.array([face[2], face[3], face[0]]))
#             print(len(faces))

    file.close()

    faces = np.array(faces) - 1
    vertices = np.array(vertices).astype('float64')
    soft_blues = np.array(
        ["#4184A4", "#3BB2E2", "#4DC8E9", "#8AE3D7", "#61D4D4"], dtype='object')
    np.random.seed(0)
    colors = np.random.choice(soft_blues, len(faces))

    return vertices, faces, colors

# inserts virtual object into an image


def insertObject(vertices, faces, colors, calMatrix, distCoeffs, image, board):

    imgpoints, objpoints, tagIDs = detect_aprilboard(image, board, at_detector)
    P = np.hstack((calMatrix, np.zeros((3, 1))))
    plane_fine = computePlane(calMatrix, distCoeffs, image, board)

    # compute normalized image coordinates (equivalent to K^{-1}*x)
    imgpts_norm = cv2.undistortPoints(imgpoints, calMatrix, distCoeffs)
    # remove extraneous size-1 dimension from openCV (annoying)
    imgpts_norm = np.squeeze(imgpts_norm)

    # back-projections are homogeneous versions of these
    backproj_fine = in2hom(imgpts_norm)

    normal_fine = np.reshape(plane_fine[0:3], (3, 1))
    d_fine = plane_fine[3]
    prod_fine = backproj_fine@normal_fine
    lam_fine = -d_fine/prod_fine
    intersect_fine = lam_fine*backproj_fine

    if len(board) == 80:
        print("fine")
        p1 = 44
        p2 = 43
        p3 = 34
        p4 = 33

    elif len(board) == 35:
        print("coarse")
        p1 = 24
        p2 = 23
        p3 = 17
        p4 = 16

    # scale
    S = np.array([[np.linalg.norm(intersect_fine[p1] - intersect_fine[p3]), 0, 0],
                  [0, np.linalg.norm(intersect_fine[p4] -
                                     intersect_fine[p3]), 0],
                  [0, 0, np.linalg.norm(intersect_fine[p2] - intersect_fine[p4])]])

    # rotation
    x = (intersect_fine[p4] - intersect_fine[p2]) / \
        (np.linalg.norm(intersect_fine[p4] - intersect_fine[p2]))
    y = (intersect_fine[p4] - intersect_fine[p3]) / \
        (np.linalg.norm(intersect_fine[p4] - intersect_fine[p3]))
    z = (np.cross(x, y))/(np.linalg.norm(np.cross(x, y)))
    R = np.array((x, y, z)).T

    # combine scale and rotation and apply
    M = np.dot(S, R)

    # translation
    center = (intersect_fine[p1] + intersect_fine[p2] +
              intersect_fine[p3] + intersect_fine[p4])/4

    vertices = np.array([np.dot(M, x) for x in vertices]) + center

    vertices = np.concatenate(
        [vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)], axis=1)
    vertices_left = np.array([np.dot(P, x) for x in vertices])
    vertices_left_in = hom2in(vertices_left)

    # plot
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')

    for i in range(len(faces)):
        plt.fill(vertices_left_in[faces[i, :], 0],     # projected x-coords for this face
                 # projected y-coords for this face
                 vertices_left_in[faces[i, :], 1],
                 color=colors[i],               # color for this face
                 alpha=0.3)                    # transparency: 1=opaque; 0=invisible

    plt.axis('off')

    return fig


def insertObjectRGB(vertices, faces, colors, calMatrix, distCoeffs, imageR, imageG, imageB, board):

    imgpoints, objpoints, tagIDs = detect_aprilboard(
        imageR, board, at_detector)
    P = np.hstack((calMatrix, np.zeros((3, 1))))
    plane_fine = computePlane(calMatrix, distCoeffs, imageR, board)

    # compute normalized image coordinates (equivalent to K^{-1}*x)
    imgpts_norm = cv2.undistortPoints(imgpoints, calMatrix, distCoeffs)
    # remove extraneous size-1 dimension from openCV (annoying)
    imgpts_norm = np.squeeze(imgpts_norm)

    # back-projections are homogeneous versions of these
    backproj_fine = in2hom(imgpts_norm)

    normal_fine = np.reshape(plane_fine[0:3], (3, 1))
    d_fine = plane_fine[3]
    prod_fine = backproj_fine@normal_fine
    lam_fine = -d_fine/prod_fine
    intersect_fine = lam_fine*backproj_fine

    if len(board) == 80:
        p1 = 44
        p2 = 43
        p3 = 34
        p4 = 33

    elif len(board) == 35:
        p1 = 24
        p2 = 23
        p3 = 17
        p4 = 16

    # plot
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(121)

    image = RGB2Color(imageR, imageG, imageB)
    plt.imshow(image)

    if (len(intersect_fine) == len(board)):
        # scale
        S = np.array([[np.linalg.norm(intersect_fine[p1] - intersect_fine[p3]), 0, 0],
                      [0, np.linalg.norm(
                          intersect_fine[p4] - intersect_fine[p3]), 0],
                      [0, 0, np.linalg.norm(intersect_fine[p2] - intersect_fine[p4])]])

        # rotation
        x = (intersect_fine[p4] - intersect_fine[p2]) / \
            (np.linalg.norm(intersect_fine[p4] - intersect_fine[p2]))
        y = (intersect_fine[p4] - intersect_fine[p3]) / \
            (np.linalg.norm(intersect_fine[p4] - intersect_fine[p3]))
        z = (np.cross(x, y))/(np.linalg.norm(np.cross(x, y)))
        R = np.array((x, y, z)).T

        # combine scale and rotation and apply
        M = np.dot(S, R)

        # translation
        center = (intersect_fine[p1] + intersect_fine[p2] +
                  intersect_fine[p3] + intersect_fine[p4])/4

        vertices = np.array([np.dot(M, x) for x in vertices]) + center

        vertices = np.concatenate(
            [vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)], axis=1)
        vertices_left = np.array([np.dot(P, x) for x in vertices])
        vertices_left_in = hom2in(vertices_left)

        for i in range(len(faces)):
            plt.fill(vertices_left_in[faces[i, :], 0],     # projected x-coords for this face
                     # projected y-coords for this face
                     vertices_left_in[faces[i, :], 1],
                     color=colors[i],               # color for this face
                     alpha=0.3)                    # transparency: 1=opaque; 0=invisible

    plt.axis('off')

    return fig


# returns the video in an array of B/W frames
def processVideo(src):
    VIDEO = cv2.VideoCapture(src)
    success, image = VIDEO.read()

    # first frame
    video = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # scale
    scale_percent = 50  # percent of original size
    width = int(video.shape[1] * scale_percent / 100)
    height = int(video.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    video = cv2.resize(video, dim, interpolation=cv2.INTER_AREA)

    # remaining frames
    while success:
        success, image = VIDEO.read()
        if success:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            video = np.dstack(
                (video, cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)))

    video = np.float32(video)/255
    video = np.uint8(video*255)

    VIDEO.release()

    return video

# returns 3 arrays of the video in frames (R, G, B)


def processVideoRGB(src):
    VIDEO = cv2.VideoCapture(src)
    success, image = VIDEO.read()

    # separate into R, G, B channels
    video = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R = video[:, :, 0]
    G = video[:, :, 1]
    B = video[:, :, 2]

    # scale
    scale_percent = 50  # percent of original size
    width = int(R.shape[1] * scale_percent / 100)
    height = int(R.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    videoR = cv2.resize(R, dim, interpolation=cv2.INTER_AREA)
    videoG = cv2.resize(G, dim, interpolation=cv2.INTER_AREA)
    videoB = cv2.resize(B, dim, interpolation=cv2.INTER_AREA)

    # remaining frames
    while success:
        success, image = VIDEO.read()
        if success:
            video = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            R = video[:, :, 0]
            G = video[:, :, 1]
            B = video[:, :, 2]
            videoR = np.dstack(
                (videoR, cv2.resize(R, dim, interpolation=cv2.INTER_AREA)))
            videoG = np.dstack(
                (videoG, cv2.resize(G, dim, interpolation=cv2.INTER_AREA)))
            videoB = np.dstack(
                (videoB, cv2.resize(B, dim, interpolation=cv2.INTER_AREA)))

    videoR = np.float32(videoR)/255
    videoR = np.uint8(videoR*255)

    videoG = np.float32(videoG)/255
    videoG = np.uint8(videoG*255)

    videoB = np.float32(videoB)/255
    videoB = np.uint8(videoB*255)

    VIDEO.release()

    return videoR, videoG, videoB

# generates a GIF for a stationary object


def generateGIF(output_filename, numframes, vertices, faces, colors, video, board):
    print("Processing video...")
    video = processVideo(video_src)

    print("Calibrating camera...")
    reprojerr, calMatrix, distCoeffs, calRotations, calTranslations = calibrateCamera(
        video, board)

    print("Creating frames...")
    images = []
    for i in tqdm(range(numframes)):
        image = video[:, :, i]

        # generate plot
        fig = insertObject(vertices, faces, colors,
                           calMatrix, distCoeffs, image, board)

        # convert plot to image array
        # various solutions available at https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
        dpi = 180  # set image resolution (not sure if this works)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(np.uint8(img))
        images.append(img)
        plt.close()

    print("Saving as GIF...")
    images[0].save(output_filename,
                   save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)

    print("Your file", output_filename, "has been saved!")

# generates a GIF for moving object, B/W background


def generateGIF2(output_filename, numframes, src_directory, video_src, board):
    print("Processing video...")
    video = processVideo(video_src)

    print("Calibrating camera...")
    reprojerr, calMatrix, distCoeffs, calRotations, calTranslations = calibrateCamera(
        video, board)

    images = []
    for i in tqdm(range(numframes)):
        image = video[:, :, i]
        obj_src = src_directory + "{:06d}".format(i+1) + ".obj"
        v, f, c = getObject(obj_src)

        # generate plot
        fig = insertObject(v, f, c, calMatrix, distCoeffs, image, board)

        # convert plot to image array
        # various solutions available at https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
        dpi = 180  # set image resolution (not sure if this works)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(np.uint8(img))
        images.append(img)
        plt.close()

    # save images as gif
    print("Saving as GIF...")
    images[0].save(output_filename,
                   save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)

    print("Your file", output_filename, "has been saved!")

# generates a GIF for a moving object, color background


def generateGIF3(output_filename, numframes, src_directory, video_src, board):
    print("Processing video...")
    vidR, vidG, vidB = processVideoRGB(video_src)

    print("Calibrating camera...")
    reprojerr, calMatrix, distCoeffs, calRotations, calTranslations = calibrateCamera(
        vidR, board)

    images = []
    print("Drawing frames...")
    for i in tqdm(range(numframes)):
        imgR = vidR[:, :, i]
        imgG = vidG[:, :, i]
        imgB = vidB[:, :, i]

        obj_src = src_directory + "{:06d}".format(i+1) + ".obj"
        v, f, c = getObject(obj_src)

        # generate plot
        fig = insertObjectRGB(v, f, c, calMatrix,
                              distCoeffs, imgR, imgG, imgB, board)

        # convert plot to image array
        # various solutions available at https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
        dpi = 180  # set image resolution (not sure if this works)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(np.uint8(img))
        images.append(img)
        plt.close()

    print("Saving as GIF...")
    images[0].save(output_filename,
                   save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)

    print("Your file", output_filename, "has been saved!")
