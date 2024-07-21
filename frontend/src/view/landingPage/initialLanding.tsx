import React from "react";
import { Button } from "antd";
import { useNavigate } from "react-router-dom";
import "./styles/initialLanding.scss";

const InitialLanding = ({ scrollToLandingStat }) => {
  const navigate = useNavigate();
  return (
    <div>
      <div className="initialLandingWrapper">
        <div className="initialLeftWrapper">
          <div className="initialLeftWrapperInner">
            <div className="initialLeftWrapperTitle">
              당신의 경험이 빛날 수 있도록
              <br />
            </div>
            <div className="initialLeftWrapperSubtitle">
              <p>
                막막한 자기소개서 작성, 이젠 AI 자소서 컨설턴트
                <br /> REDITOR가 함께하겠습니다.
              </p>
            </div>
            <div className="initialButtonWrapper">
              <Button
                className="initialLoginBtn"
                size="large"
                onClick={() => {
                  navigate("/auth/login");
                }}
              >
                REDITOR 로그인하기
              </Button>
              <Button
                className="initialSignUpBtn"
                size="large"
                onClick={() => {
                  navigate("/auth/signup");
                }}
              >
                회원가입 먼저하기
              </Button>
            </div>
          </div>
        </div>
        <div className="initialRightWrapper">
          <img
            className="initialPageImage"
            src="/img/logo.gif"
            alt="이미지를 표시할수 없습니다"
          />
        </div>
      </div>
      <div className="moreInfoBtnWrapper">
        <Button
          className="moreInfoBtn"
          size="large"
          onClick={scrollToLandingStat}
        >
          <span>더 알아보기</span>
        </Button>
      </div>
    </div>
  );
};

export default InitialLanding;
