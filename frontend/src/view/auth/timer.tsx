import { RootState } from "@/store/store";
import { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { stop } from "@/store/features/timer/timerSlice";
import React from "react";

const Timer = (props) => {
  const [seconds, setSeconds] = useState(300); //seconds
  const timer = useSelector((state: RootState) => state.timer.value);
  const dispatch = useDispatch();

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    if (timer) {
      // 1초마다 콜백함수 실행
      intervalId = setInterval(() => {
        setSeconds((prevSeconds) => {
          if (prevSeconds <= 0) {
            dispatch(stop());
            return 300;
          }
          return prevSeconds - 1;
        });
      }, 1000);
    }

    // clean-up 함수에서 interval을 취소
    return () => clearInterval(intervalId);
  }, [timer]);

  // 시간을 분:초로 변환
  const formatTime = (time: number): string => {
    const minutes = Math.floor(time / 60);
    const seconds = time % 60;
    return `${minutes.toString().padStart(2, "0")}:${seconds
      .toString()
      .padStart(2, "0")}`;
  };

  return <b> {formatTime(seconds)}</b>;
};
export default Timer;
