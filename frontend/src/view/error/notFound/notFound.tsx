import React from "react";

const NotFound = () => {
  return (
    <div>
      <div
        style={{
          width: "100vw",
          height: "100vh",
          display: "flex",
          justifyContent: "center",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <div>
          <h1>404 - Not Found</h1>
        </div>
        <div style={{ textAlign: "center" }}>
          <p>
            죄송합니다. 요청하신 페이지를 찾을 수 없습니다.
            <br /> <br />
            주소를 확인해 주세요.
          </p>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
