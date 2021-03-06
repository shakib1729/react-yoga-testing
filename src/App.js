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
  const [timerVar, setTimerVar] = useState(null);
  const [poseNetModel, setPoseNetModel] = useState(null);
  const [model, setModel] = useState(null);
  const [timer, setTimer] = useState(10);
  const [currPose, setCurrPose] = useState(0);
  const [scores, setScores] = useState([0, 0]);

  const timerRef = useRef(timer);
  timerRef.current = timer;

  const currPoseRef = useRef(currPose);
  currPoseRef.current = currPose;

  const isDetectingRef = useRef(isDetecting);
  isDetectingRef.current = isDetecting;

  const intervalVarRef = useRef(intervalVar);
  intervalVarRef.current = intervalVar;

  const timerVarRef = useRef(timerVar);
  timerVarRef.current = timerVar;

  const scoresRef = useRef(scores);
  scoresRef.current = scores;

  useEffect(() => {
    const loadModels = async () => {
      // Load the posenet model
      const loadedPoseNetModel = await posenet.load({
        inputResolution: { width: 426, height: 240 },
        scale: 0.5,
      });
      setPoseNetModel(loadedPoseNetModel);

      // Load the trained neural network model
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

    // Stop detecting after 10 seconds
    setTimer(10);
    const timerVar = setInterval(() => {
      setTimer(timerRef.current - 1);
      if (timerRef.current == 0) {
        clearInterval(timerVar);
        changeIsDetecting();
      }
    }, 1000);
    setTimerVar(timerVar);
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
    console.log(
      'Predicted: ',
      predictedClass[0],
      ' Current:  ',
      currPoseRef.current
    );
    if (predictedClass[0] == currPoseRef.current) {
      // We made a correct prediction,
      // so increment the score of current pose

      setScores(
        scoresRef.current.map((currScore, idx) => {
          if (idx == currPoseRef.current) return currScore + 1;
          else return currScore;
        })
      );
      console.log('Yes');
    } else {
      console.log('No');
    }
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

  const resetStates = () => {
    setIsDetecting(false);
    clearInterval(intervalVarRef.current);
    clearInterval(timerVarRef.current);
    setTimer(10);
  };

  const startDetecting = (pose) => {
    setCurrPose(pose);
    setIsDetecting(true);
    const timerVar = setInterval(() => {
      setTimer(timerRef.current - 1);
      if (timerRef.current == 0) {
        clearInterval(timerVar);
        runPosenet();
      }
    }, 1000);
    setTimerVar(timerVar);
  };
  const changeIsDetecting = (pose) => {
    if (isDetectingRef.current) resetStates();
    else startDetecting(pose);
  };

  return (
    <div className='App'>
      Current Pose: {currPose}
      <button onClick={() => changeIsDetecting(0)}>
        <h4> {isDetecting ? 'Stop detecting' : 'Detect pose 0'} </h4>
      </button>
      Timer: {timer}
      <button onClick={() => changeIsDetecting(1)}>
        <h4> {isDetecting ? 'Stop detecting' : 'Detect pose 1'} </h4>
      </button>
      {scores.map((score, idx) => {
        return (
          <li key={idx}>
            Pose {idx}: {score}
          </li>
        );
      })}
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
