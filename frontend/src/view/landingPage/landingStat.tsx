import axios from "axios";
import React, { CSSProperties, useEffect, useState } from "react";
import CountUp from "react-countup";
import { useInView } from "react-intersection-observer";
import "./styles/landingStat.scss";

const LandingStat = () => {
  const [totalUser, setTotalUser] = useState<number>(0);
  const [totalVisitor, setTotalVisitor] = useState<number>(0);
  const [totalResume, setTotalResume] = useState<number>(0);
  const [totalEdit, setTotalEdit] = useState<number>(0);

  const { ref, inView } = useInView({
    triggerOnce: false,
    threshold: 0.1,
  });

  const animationDuration = 4;

  const fetchLandingStatData = async (group: string) => {
    let res = await axios.get(`/landing/statistics/${group}`).then((res) => {
      switch (group) {
        case "user-count":
          setTotalUser(res.data.total_count);
          break;
        case "total-visit":
          setTotalVisitor(res.data.visit_total);
          break;
        case "total-edit":
          setTotalEdit(res.data.total_edit);
          break;
        case "total-board":
          setTotalResume(res.data.total_board);
          break;
      }
    });
  };

  useEffect(() => {
    Promise.all([
      fetchLandingStatData("user-count"),
      fetchLandingStatData("total-visit"),
      fetchLandingStatData("total-edit"),
      fetchLandingStatData("total-board"),
    ]);
  }, []);

  return (
    <div className="landingStatWrapper" ref={ref}>
      <div className="landingStatHeader">
        <div className="landingStatTitle">당신만을 위한 AI 자소서 컨설턴트</div>
        <div className="landingStatSubTitle">
          <p>
            수많은 취업 준비생들과 이직 준비생들이 Reditor로 자기소개서를
            첨삭하고 있어요.
          </p>
        </div>
      </div>
      <div className="statWrapper">
        <div className="statInnerWrapper">
          <div className="statDataWrapper">
            <div className="dataStyle">
              <div>
                <img
                  width={50}
                  height={50}
                  src="/img/xo.png"
                  alt="이미지를 표시할수 없습니다!"
                />
              </div>
              <div className="dataHeaderStyle">첨삭 횟수</div>
              {inView && (
                <CountUp
                  start={0}
                  end={totalEdit}
                  duration={animationDuration}
                />
              )}
              회
            </div>
            <div className="dataStyle">
              <img
                width={50}
                height={50}
                src="/img/human.png"
                alt="이미지를 표시할수 없습니다!"
              />
              <div className="dataHeaderStyle">가입한 유저</div>
              {inView && (
                <CountUp
                  start={0}
                  end={totalUser}
                  duration={animationDuration}
                />
              )}
              명
            </div>
            <div className="dataStyle">
              <img
                width={50}
                height={50}
                src="/img/memo.png"
                alt="이미지를 표시할수 없습니다!"
              />
              <div className="dataHeaderStyle">저장된 자소서</div>
              {inView && (
                <CountUp
                  start={0}
                  end={totalResume}
                  duration={animationDuration}
                />
              )}
              개
            </div>
            <div className="dataStyle">
              <img
                width={50}
                height={50}
                src="/img/graph.png"
                alt="이미지를 표시할수 없습니다!"
              />
              <div className="dataHeaderStyle">총 방문자</div>
              {inView && (
                <CountUp
                  start={0}
                  end={totalVisitor}
                  duration={animationDuration}
                />
              )}
              명
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingStat;
