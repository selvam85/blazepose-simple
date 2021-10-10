import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';
import '@mediapipe/pose';

let video = document.getElementById('video');
let canvas = document.getElementById('output');
let ctx = canvas.getContext('2d');
let detector, model;

const COLOR_PALETTE = [
    '#ffffff', '#800000', '#469990', '#e6194b', '#42d4f4', '#fabed4', '#aaffc3',
    '#9a6324', '#000075', '#f58231', '#4363d8', '#ffd8b1', '#dcbeff', '#808000',
    '#ffe119', '#911eb4', '#bfef45', '#f032e6', '#3cb44b', '#a9a9a9'
];

async function createDetector() {
    model = poseDetection.SupportedModels.BlazePose;
    const detectorConfig = {
        runtime: "tfjs",
        enableSmoothing: true,
        modelType: "full"
    };
    detector = await poseDetection.createDetector(model, detectorConfig);
}

async function setupCamera() {
    if(navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({video: true})
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(e => {
                console.log("Error occurred while getting the video stream");
            });
    }
    
    /* video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
    }; */

    await new Promise((resolve) => {
        video.onloadedmetadata = () => {
          resolve(video);
        };
    });

    video.play();

    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    // Must set below two lines, otherwise video element doesn't show.
    video.width = videoWidth;
    video.height = videoHeight;

    canvas.width = videoWidth;
    canvas.height = videoHeight;
    const canvasContainer = document.querySelector('.canvas-wrapper');
    canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

    /* canvas.width = video.videoWidth;
    canvas.height = video.videoHeight; */

    // Because the image from camera is mirrored, need to flip horizontally.
    ctx.translate(video.videoWidth, 0);
    ctx.scale(-1, 1);
}

async function predictPoses() {
    await renderResult();
    window.requestAnimationFrame(predictPoses);
};

async function renderResult() {
    if (video.readyState < 2) {
        await new Promise((resolve) => {
          video.onloadeddata = () => {
            resolve(video);
          };
        });
    }

    let poses = null;

    // Detector can be null if initialization failed (for example when loading
    // from a URL that does not exist).
    if (detector != null) {
        // Detectors can throw errors, for example when using custom URLs that
        // contain a model that doesn't provide the expected output.
        try {
            poses = await detector.estimatePoses(video, { 
                flipHorizontal: false 
            });
        } catch (error) {
            detector.dispose();
            detector = null;
            alert(error);
        }
    }

    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    if (poses && poses.length > 0) {
        for (const pose of poses) {
            if (pose.keypoints != null) {
                drawKeypoints(pose.keypoints);
                drawSkeleton(pose.keypoints, pose.id);
            }
        }
    }
}

function drawKeypoints(keypoints) {
    const keypointInd = poseDetection.util.getKeypointIndexBySide(model);
    ctx.strokeStyle = 'White';
    ctx.lineWidth = 2;

    ctx.fillStyle = 'Red';
    for (const i of keypointInd.middle) {
        drawKeypoint(keypoints[i]);
    }

    ctx.fillStyle = 'Green';
    for (const i of keypointInd.left) {
        drawKeypoint(keypoints[i]);
    }

    ctx.fillStyle = 'Orange';
    for (const i of keypointInd.right) {
        drawKeypoint(keypoints[i]);
    }
}

function drawKeypoint(keypoint) {
    // If score is null, just show the keypoint.
    const score = keypoint.score != null ? keypoint.score : 1;
    const scoreThreshold = 0.6;
    const radius = 4;

    if (score >= scoreThreshold) {
      const circle = new Path2D();
      circle.arc(keypoint.x, keypoint.y, radius, 0, 2 * Math.PI);
      ctx.fill(circle);
      ctx.stroke(circle);
    }
}

function drawSkeleton(keypoints, poseId) {
    // Each poseId is mapped to a color in the color palette.
    const color = COLOR_PALETTE[poseId % 20];
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;

    poseDetection.util.getAdjacentPairs(model)
        .forEach(([i, j]) => {
            const kp1 = keypoints[i];
            const kp2 = keypoints[j];

            // If score is null, just show the keypoint.
            const score1 = kp1.score != null ? kp1.score : 1;
            const score2 = kp2.score != null ? kp2.score : 1;
            const scoreThreshold = 0.6;

            if (score1 >= scoreThreshold && score2 >= scoreThreshold) {
                ctx.beginPath();
                ctx.moveTo(kp1.x, kp1.y);
                ctx.lineTo(kp2.x, kp2.y);
                ctx.stroke();
            }
    });
}

async function app() {
    await createDetector();
    await setupCamera();
    predictPoses();
};

app();
