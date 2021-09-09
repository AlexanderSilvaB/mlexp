let video;

let posenet;
let classifier;
let facemesh;
let objectDetector;
let handpose;

let specs;

let poses = [];
let classify = [];
let faces = [];
let objects = [];
var hands = [];

let enablePoseNet = true;
let enableClassifier = false;
let enableFacemesh = false;
let enableObjectDetector = false;
let enableHandPose = false;

function setup() {  // this function runs only once while running
    createCanvas(800, 500);
    //console.log("setup funct");
    video = createCapture(VIDEO, setupML);
    video.hide();
    // assets
    specs = loadImage('spects.png');
}

function setupML()
{
    if(enablePoseNet)
    {
        posenet = ml5.poseNet(video, {
            inputResolution : 161
        }, function()
        {
            console.log('poseNet loaded!');
        });
        posenet.on('pose', onPoses);
    }

    if(enableClassifier)
    {
        classifier = ml5.imageClassifier('MobileNet', video, function(){
            console.log('imageClassifier loaded!');
            onClassify(null, []);
        });
    }

    if(enableFacemesh)
    {
        facemesh = ml5.facemesh(video, function()
        {
            console.log('facemesh loaded!');
        });
        facemesh.on("predict", onFaces);
    }

    if(enableObjectDetector)
    {
        objectDetector = ml5.objectDetector('cocossd', video, function(){
            console.log('objectDetector loaded!');
            onDetect(null, []);
        });
    }

    if(enableHandPose)
    {
        handpose = ml5.handpose(video, function()
        {
            console.log('handpose loaded!');
        });
        handpose.on("predict", onHands);
    }
}

function onPoses(p) {
    poses = p;
}

function onFaces(f)
{
    if(f.length > 0)
        faces = f;
}

function onClassify(err, results) {
    classify = results;
    classifier.classify(onClassify);
}

function onDetect(err, results) {
    objects = results;
    objectDetector.detect(onDetect);
}

function onHands(h)
{
    hands = h;
}


function draw() { // this function code runs in infinite loop
    
    // images and video(webcam)
    image(video, 0, 0);
    
    // Skeleton
    var singlePose;
    var skeleton;
    fill(255, 0, 0);
    stroke(0);
    for(var i = 0; i < poses.length; i++)
    {
        singlePose = poses[i].pose;
        skeleton = poses[i].skeleton;
        for(let i=0; i<singlePose.keypoints.length; i++) {
            if(singlePose.keypoints[i].score > 0.1)
            {
                ellipse(singlePose.keypoints[i].position.x, singlePose.keypoints[i].position.y, 2);
            }
        }

        stroke(255, 255, 255);
        strokeWeight(1);

        for(let j=0; j<skeleton.length; j++) {
            line(skeleton[j][0].position.x, skeleton[j][0].position.y, skeleton[j][1].position.x, skeleton[j][1].position.y);
        }

        // Apply specs and cigar
        image(specs, singlePose.leftEye.x + 45, singlePose.leftEye.y - 55, (singlePose.rightEye.x - singlePose.leftEye.x) * 2.2, 150);
    }

    // Classify
    fill(255, 0, 0);
    stroke(0);
    for(var i = 0; i < classify.length; i++)
    {
        text(classify[i].confidence.toFixed(2) + " - " + classify[i].label, 20, 40 + 20 * i);
    }

    // Face
    fill(0, 255, 0);
    stroke(0);
    for(var i = 0; i < faces.length; i += 10)
    {
        for(var j = 0; j < faces[i].scaledMesh.length; j++)
        {
            ellipse(faces[i].scaledMesh[j][0], faces[i].scaledMesh[j][1], 4);
        }
    }

    // Objects
    noFill();
    stroke(0, 255, 0);
    for (var i = 0; i < objects.length; i++) 
    {
        text(objects[i].label, objects[i].x + 4, objects[i].y + 16);
        rect(objects[i].x, objects[i].y, objects[i].width, objects[i].height);
    }

    // Hands
    fill(0, 0, 255);
    stroke(0);
    for(var i = 0; i < hands.length; i += 10)
    {
        for(var j = 0; j < hands[i].landmarks.length; j++)
        {
            ellipse(hands[i].landmarks[j][0], hands[i].landmarks[j][1], 4);
        }
    }
}