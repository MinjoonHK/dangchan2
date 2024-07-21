import React, { CSSProperties } from "react";
import type { CollapseProps } from "antd";
import { Collapse } from "antd";

const fontStyle: CSSProperties = {
  fontSize: "1.3rem",
  fontWeight: "bold",
};

const itemStyle: CSSProperties = {
  width: "70%",
};
const items: CollapseProps["items"] = [
  {
    key: "1",
    label: <div style={fontStyle}>Reditor란?</div>,
    children: (
      <div>
        <p>세계에서 가장 정밀한 자기소개서 첨삭 서비스를 제공합니다.</p>
        <p>
          20,000개의 합격 자기소개서 데이터를 기반으로 정제된 첨삭 기능을
          제공하며, PRO 모드를 사용하면 더 좋은 성능의 결과와 다양한 기법을
          활용한 첨삭이 진행됩니다.
        </p>
        <p>
          1. 자기소개서 작성이 막막한 취업 준비생을 위한 자기소개서 작성 가이드
          기능을 제공합니다.
        </p>
        <p>2. 최고의 자기소개서 첨삭 서비스를 무료로 제공합니다.</p>
        <p>
          3. Reditor 유저들이 첨삭 받은 자기소개서를 열람할 수 있는 게시판을
          제공합니다. 지원하고자 하는 회사/직무의 자기소개서를 참고해보세요.
        </p>
      </div>
    ),
  },
  {
    key: "2",
    label: <div style={fontStyle}>자기소개서 첨삭 이용방법</div>,
    children: (
      <div>
        <p>1. 상단 메뉴에서 [자소서 첨삭]을 클릭합니다.</p>
        <p>
          <img
            style={itemStyle}
            src="/img/faq_navbar.png"
            alt="이미지를 표시할수 없습니다"
          />
        </p>
        <p>
          2. 신입/경력 선택, pro/lite 모드 선택, 지원 회사, 지원 직무, 첨삭 받을
          자기소개서의 문항, 첨삭을 원하는 자기소개서 내용을 작성합니다.
        </p>
        <p>
          <img
            style={itemStyle}
            src="/img/faq_edit.png"
            alt="이미지를 표시할수 없습니다"
          />
        </p>
        <p>
          3. [자소서 첨삭하기]를 클릭하면 작성된 내용을 바탕으로 Reditor가
          첨삭을 시작합니다. 오른쪽 화면에서 완료된 첨삭 결과를 확인할 수
          있으며, [Diff 결과 보기]를 눌러 첨삭된 부분을 상세하게 확인할 수
          있습니다.
        </p>
        <p>
          <img
            style={itemStyle}
            src="/img/edit_result.gif"
            alt="이미지를 표시할수 없습니다"
          />
        </p>
      </div>
    ),
  },
  {
    key: "3",
    label: <div style={fontStyle}>자기소개서 가이드 이용 방법</div>,
    children: (
      <div>
        <p>1. 상단 메뉴에서 [자소서 가이드] 클릭합니다.</p>
        <p>
          <img
            style={itemStyle}
            src="/img/faq_guide_navbar.png"
            alt="이미지를 표시할수 없습니다"
          />
        </p>
        <p>
          2. 신입/경력 선택, 지원 회사, 지원 직무, 첨삭 받을 자기소개서의 문항,
          자기소개서에 작성할 본인의 수상 경력과 직무 관련 경험을 작성합니다.
        </p>
        <p>
          <img
            style={itemStyle}
            src="/img/faq_guide_usage.gif"
            alt="이미지를 표시할수 없습니다"
          />
        </p>
        <p>
          3. [생성하기]를 클릭하면 작성된 내용을 바탕으로 Reditor가 가이드
          작성을 시작합니다. 오른쪽 화면에서 작성된 가이드를 확인할 수 있습니다.
        </p>
        <p>
          <img
            style={itemStyle}
            src="/img/faq_guide_result.gif"
            alt="이미지를 표시할수 없습니다"
          />
        </p>
      </div>
    ),
  },
  {
    key: "4",
    label: <div style={fontStyle}>PRO/LITE 모드의 차이가 궁금해요</div>,
    children: (
      <div>
        <p>
          pro 모드는 첨삭 결과 자기소개서가 자소서 게시판에 공유됩니다. 대신 더
          정교한 첨삭이 이루어지고, 원하는 기법이 적용된 첨삭을 받아볼 수
          있어요.
        </p>
        <p>lite 모드는 기본 첨삭이 이루어지며 게시판에 공유되지 않습니다.</p>
        <p>
          <img
            style={itemStyle}
            src="/img/pro_lite_description.gif"
            alt="이미지를 표시할수 없습니다"
          />
        </p>
      </div>
    ),
  },
];

const FAQ = () => {
  return (
    <div style={{ display: "flex", justifyContent: "center" }}>
      <div style={{ width: "80%", marginTop: "5%", marginBottom: "12%" }}>
        <div
          style={{
            width: "100%",
            textAlign: "center",
            fontWeight: "bold",
            fontSize: "1.5rem",
            textDecoration: "underline",
            marginBottom: "5%",
          }}
        >
          Frequently Asked Question
        </div>
        <Collapse
          size="large"
          style={{ backgroundColor: "white" }}
          accordion
          items={items}
        />
      </div>
    </div>
  );
};

export default FAQ;
