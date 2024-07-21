import { CSSProperties } from "react";
import BounceLoader from "react-spinners/BounceLoader";
import FadeLoader from "react-spinners/FadeLoader";
import PacmanLoader from "react-spinners/PacmanLoader";
import PuffLoader from "react-spinners/PuffLoader";

const LoadingSpinner = () => {
    const descriptionStyle: CSSProperties = {
      textAlign: 'center',
      marginTop: '3%',
    };

    const tipStyle: CSSProperties = {
      fontSize: '0.8rem',
      marginTop: '5%',
    };
    let spinner = [
      <div>
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <PacmanLoader color="#36d7b7" />
        </div>
        <div style={descriptionStyle}>
          가이드받은 내용을 토대로 나만의 자기소개서를 만들어보세요
        </div>
        <div style={tipStyle}>
          Tip. 자기소개서에는 특별한 경험을 녹여낼 수록 좋아요.{' '}
        </div>
      </div>,
      <div>
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <PuffLoader color="#36d7b7" />
        </div>
        <div style={descriptionStyle}>
          Reditor는 Resume Editor의 줄임말입니다!
        </div>
        <div style={tipStyle}>
          Tip. 자소서를 쓰기 전에 내세울 수 있는 나의 경험을 정리하면
          <br />
          훨씬 좋은 자소서를 쓸 수 있어요!
        </div>
      </div>,
      <div>
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <BounceLoader color="#36d7b7" />
        </div>
        <div style={descriptionStyle}>
          수 만개의 자기소개서를 기반으로 가이드를 생성 중입니다!
        </div>
        <div style={tipStyle}>
          Tip. 추상적인 표현보다는 명료한 표현이 좋아요!
        </div>
      </div>,
      <div>
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <FadeLoader color="#36d7b7" />
        </div>
        <div style={descriptionStyle}>
          Reditor에 등록된 높은 평점의 자기소개서는
          <br /> 다시 자기소개서 첨삭AI에 활용돼요.
        </div>
        <div style={tipStyle}>
          Tip. 자기소개서 목록에서 다른 사람들의 자기소개서도 참고해보세요!
        </div>
      </div>,
    ];

    let randomIndex = Math.floor(Math.random() * spinner.length);

    return spinner[randomIndex];
  };

export default LoadingSpinner;