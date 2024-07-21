import React from "react";
import Marquee from "react-fast-marquee";
import ReviewData from "./reviewData";
import CustomFooter from "@/components/footer";
import "./styles/landingReview.scss";

const LandingReview = () => {
  return (
    <div className="landingReviewWrapper">
      <div className="landingReviewInnerWrapper">
        <div className="landingReviewHeader">실제 사용자들의 생생한 후기!</div>
        <div className="landingReviewSubHeader">
          20,000개의 합격 자기소개서 데이터와 AI의 결합으로 취준생들이 반한
          자기소개서 첨삭 전문 서비스!
        </div>
        <div className="landingReviewMarqueeWrapper">
          <Marquee pauseOnHover={true}>
            {ReviewData.map((data, index) => {
              return (
                <ReviewBox
                  key={`${data} ${index}`}
                  username={data.username}
                  comment={data.comment}
                  avatar={data.avatar}
                />
              );
            })}
          </Marquee>
          <Marquee
            pauseOnHover={true}
            direction="right"
            style={{ marginTop: "5%" }}
          >
            {ReviewData.map((data, index) => {
              return (
                <ReviewBox
                  key={`${data}2 ${index}`}
                  username={data.username}
                  comment={data.comment}
                  avatar={data.avatar}
                />
              );
            })}
          </Marquee>
        </div>
      </div>
      <div className="landingFooter">
        <CustomFooter />
      </div>
    </div>
  );
};

const ReviewBox = ({ username, comment, avatar }) => {
  return (
    <div className="reviewBoxWrapper">
      <h3>{username}</h3>
      <p>{comment}</p>
    </div>
  );
};

export default LandingReview;
