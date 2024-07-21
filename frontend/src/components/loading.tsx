import PacmanLoader from "react-spinners/PacmanLoader";
import React from "react";

const Loading = ({ fullscreen = false }) => {
  return (
    <div
      style={{
        width: fullscreen ? "100vw" : "100%",
        height: fullscreen ? "100vh" : "100%",
        position: fullscreen ? "fixed" : "relative",
        top: 0,
        left: 0,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexDirection: "column",
        backgroundColor: fullscreen
          ? "rgba(255, 255, 255, 0.8)"
          : "transparent",
        zIndex: fullscreen ? 1000 : "auto",
      }}
    >
      <div>
        <PacmanLoader color={"#36D7B7"} size={35} />
      </div>
      <div>로딩중입니다...</div>
    </div>
  );
};

export default Loading;
