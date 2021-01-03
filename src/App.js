import React, { useRef, useState, useEffect } from 'react';
import './App.css';

import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';
import Webcam from 'react-webcam';
import { drawKeypoints, drawSkeleton } from './utilities';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [intervalVar, setIntervalVar] = useState(null);
  const [poseNetModel, setPoseNet] = useState(null);
  const [model, setModel] = useState(null);

  useEffect(() => {
    // Load the posenet model
    const loadModels = async () => {
      const net = await posenet.load({
        inputResolution: { width: 426, height: 240 },
        scale: 0.5,
      });
      setPoseNet(net);

      // Load the trained neural network model
      // const response = await fetch('/data/Y.csv');
      const loadedModel = await tf.loadLayersModel('/model/my-model.json');
      setModel(loadedModel);
      // loadedModel.summary();
    };
    loadModels();
  }, []);

  const runPosenet = () => {
    const newInterval = setInterval(() => {
      detect();
    }, 100);
    setIntervalVar(newInterval);
  };

  const detect = async () => {
    if (
      typeof webcamRef.current !== 'undefined' &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState == 4 &&
      typeof canvasRef.current !== 'undefined' &&
      canvasRef.current !== null
    ) {
      // Get video properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width (because when we are working with webcam we need to force the height and width)
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Make detections
      const pose = await poseNetModel.estimateSinglePose(video, {
        flipHorizontal: true,
      });
      let currDataPoint = [];
      for (let i = 0; i < pose['keypoints'].length; i++) {
        currDataPoint.push(pose['keypoints'][i]['position'].x / 400);
        currDataPoint.push(pose['keypoints'][i]['position'].y / 400);
      }
      // console.log(currDataPoint);

      predict(currDataPoint);

      requestAnimationFrame(() => {
        drawCanvas(pose, videoWidth, videoHeight, canvasRef);
      });
    }
  };

  const predict = async (input) => {
    const inputTensor = tf.tensor2d(input, [1, input.length]);
    const prediction = model.predict(inputTensor);
    const predictedClass = prediction.argMax(-1).dataSync();
    console.log(predictedClass[0]);
  };

  const drawCanvas = (pose, videoWidth, videoHeight, canvas) => {
    if (canvas.current !== null) {
      const ctx = canvas.current.getContext('2d');
      canvas.current.width = videoWidth;
      canvas.current.height = videoHeight;

      drawKeypoints(pose['keypoints'], 0.5, ctx);
      drawSkeleton(pose['keypoints'], 0.5, ctx);
    }
  };

  const changeIsDetecting = () => {
    if (isDetecting) {
      setIsDetecting(false);
      clearInterval(intervalVar);
    } else {
      setIsDetecting(true);
      runPosenet();
    }
  };

  return (
    <div className='App'>
      <button onClick={changeIsDetecting}>
        <h4> {isDetecting ? 'Stop detecting' : 'Start detecting'} </h4>
      </button>
      <header className='App-header'>
        {isDetecting ? (
          <>
            <Webcam
              ref={webcamRef}
              mirrored
              style={{
                position: 'absolute',
                marginLeft: 'auto',
                marginRight: 'auto',
                left: 0,
                right: 0,
                textAlign: 'center',
                zindex: 9,
                width: 426,
                height: 240,
              }}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                marginLeft: 'auto',
                marginRight: 'auto',
                left: 0,
                right: 0,
                textAlign: 'center',
                zindex: 9,
                width: 426,
                height: 240,
              }}
            />
          </>
        ) : (
          <p>Turn On</p>
        )}
      </header>
    </div>
  );
}

export default App;
