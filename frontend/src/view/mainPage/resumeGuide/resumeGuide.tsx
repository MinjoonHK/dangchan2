import { Button, Form, Input, Radio, Tooltip, Modal, Table } from "antd";
import {
  InfoCircleOutlined,
  PlusOutlined,
  MinusOutlined,
  SearchOutlined,
  StarOutlined,
  StarFilled,
} from "@ant-design/icons";
import { CSSProperties, useEffect, useState } from "react";
import { useForm } from "antd/es/form/Form";
import axios from "axios";
import PacmanLoader from "react-spinners/PacmanLoader";
import PuffLoader from "react-spinners/PuffLoader";
import BounceLoader from "react-spinners/BounceLoader";
import FadeLoader from "react-spinners/FadeLoader";
import { DecodedToken } from "@/types/globalTypes";
import { jwtDecode } from "jwt-decode";
import Swal from "sweetalert2";
import axiosInstance from "@/api/api";
import React from "react";
import "../resumeList/details/star.css";
import TextArea from "antd/es/input/TextArea";

const ResumeGuide = () => {
  const [openRateModal, setOpenRateModal] = useState<boolean>(false);
  const [userInputForm] = useForm();
  const [generated, setGenerated] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState("");
  const [questionList, setQuestionList] = useState([
    { value: "", iconType: "plus" },
  ]);
  const [awardList, setAwardList] = useState([{ value: "", iconType: "plus" }]);
  const [experienceList, setExperienceList] = useState([
    { value: "", iconType: "plus" },
  ]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [isOccupationModalOpen, setIsOccupationModalOpen] = useState(false);
  const [occupationSearchResults, setOccupationSearchResults] = useState([]);
  const [occupationSearchLoading, setOccupationSearchLoading] = useState(false);
  const [occupationSearchError, setOccupationSearchError] = useState("");
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(0);

  const handleClick = (value) => {
    let num = parseFloat(value);
    setRating(num);
  };

  const handleMouseOver = (value) => {
    setHover(value);
  };

  const handleMouseOut = () => {
    setHover(0);
  };

  useEffect(() => {
    const accessToken = localStorage.getItem("access") ?? "";
    const decodedToken: DecodedToken = jwtDecode(accessToken);
    const uNum = decodedToken.uNum;

    axiosInstance
      .get(`/resume-guide/load/${uNum}`)
      .then((res) => {
        if (res.data.status === "Success") {
          const { awards, experiences } = res.data;
          if (awards) {
            const formattedAwards = awards.split("\n").map((award, index) => ({
              value: award,
              iconType: index === 0 ? "plus" : "minus",
            }));
            setAwardList(formattedAwards);
          }
          if (experiences) {
            const formattedExperiences = experiences
              .split("\n")
              .map((exp, index) => ({
                value: exp,
                iconType: index === 0 ? "plus" : "minus",
              }));
            setExperienceList(formattedExperiences);
          }
        } else if (res.data.status === "Error") {
          Swal.fire({
            icon: "error",
            title: "Error",
            text: res.data.message,
          });
        }
      })
      .catch((err) => {
        Swal.fire({
          icon: "error",
          title: "Network Error",
          text: "Failed to connect to the server. Please try again later.",
        });
      });
  }, []);

  const randomSpinner = () => {
    const descriptionStyle: CSSProperties = {
      textAlign: "center",
      marginTop: "3%",
    };

    const tipStyle: CSSProperties = {
      fontSize: "0.8rem",
      marginTop: "5%",
    };
    let spinner = [
      <div>
        <div style={{ display: "flex", justifyContent: "center" }}>
          <PacmanLoader color="#36d7b7" />
        </div>
        <div style={descriptionStyle}>
          가이드받은 내용을 토대로 나만의 자기소개서를 만들어보세요
        </div>
        <div style={tipStyle}>
          Tip. 자기소개서에는 특별한 경험을 녹여낼 수록 좋아요.{" "}
        </div>
      </div>,
      <div>
        <div style={{ display: "flex", justifyContent: "center" }}>
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
        <div style={{ display: "flex", justifyContent: "center" }}>
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
        <div style={{ display: "flex", justifyContent: "center" }}>
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

  const onFinish = ({ status, company, occupation }) => {
    const questions = questionList.filter((q) => q.value.trim() !== "");
    const awards = awardList.filter((a) => a.value.trim() !== "");
    const experiences = experienceList.filter((e) => e.value.trim() !== "");

    if (
      status === undefined ||
      company === undefined ||
      occupation === undefined
    ) {
      Swal.fire({
        icon: "error",
        title: "입력되지 않은 항목이 있습니다.",
        text: "모든 항목을 입력해주세요.",
      });
      return;
    }

    if (questions.length === 0) {
      Swal.fire({
        icon: "error",
        title: "입력되지 않은 항목이 있습니다.",
        text: "질문을 한 개 이상 입력해주세요.",
      });
      return;
    }

    if (awards.length === 0 && experiences.length === 0) {
      Swal.fire({
        icon: "error",
        title: "입력되지 않은 항목이 있습니다.",
        text: "수상경력/직무경험을 한 개 이상 작성해주세요.",
      });
      return;
    }

    setGenerated(true);
    setIsLoading(true);

    const formattedQuestions = questions.map((q, index) => ({
      [`question${index + 1}`]: q.value,
    }));
    const formattedAwards = awards.map((a, index) => ({
      [`award${index + 1}`]: a.value,
    }));
    const formattedExperiences = experiences.map((e, index) => ({
      [`experience${index + 1}`]: e.value,
    }));

    const accessToken = localStorage.getItem("access") ?? "";
    const decodedToken: DecodedToken = jwtDecode(accessToken);
    const uNum = decodedToken.uNum;
    const axiosInstance_python = axios.create({
      baseURL: "https://resume-editor-python.vercel.app",
      timeout: 100000,
      withCredentials: true,
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });

    axiosInstance
      .post(
        "/resume-guide/upload",
        {
          uNum,
          awards: awards.map((a) => a.value).join("\n"), // 배열을 문자열로 변환하여 전송
          experiences: experiences.map((e) => e.value).join("\n"), // 배열을 문자열로 변환하여 전송
        },
        {
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
        }
      )
      .then((uploadRes) => {
        axiosInstance_python
          .post(
            "/resume_guide",
            {
              status: status,
              company: company,
              occupation: occupation,
              questions: formattedQuestions,
              awards: formattedAwards,
              experiences: formattedExperiences,
            },
            {
              withCredentials: true,
              timeout: 200000,
            }
          )
          .then((res) => {
            if (res.data.status === "Success") {
              setIsLoading(false);
              setResult(res.data.result);
            }
          })
          .catch((err) => {
            console.log(
              company,
              occupation,
              questions,
              formattedAwards,
              formattedExperiences
            );
            console.log(err);
            setIsLoading(false);
            Swal.fire({
              icon: "error",
              title: "Network Error",
              text: "Failed to connect to the server. Please try again later.",
            });
          });
      })
      .catch((err) => {
        console.log(err);
        setIsLoading(false);
        Swal.fire({
          icon: "error",
          title: "Network Error",
          text: "Failed to connect to the server. Please try again later.",
        });
      });
  };
  const applyStyleToText = (text) => {
    // '*' 개수를 세는 정규 표현식
    const asteriskCount = (text.match(/\*/g) || []).length;
    // '#' 개수를 세는 정규 표현식
    const hashCount = (text.match(/#/g) || []).length;

    // '*' 개수에 따라 스타일 적용
    const style = {
      fontWeight: asteriskCount > 0 ? "bold" : "normal",
      fontSize: asteriskCount > 0 ? "1.1rem" : "1rem",
      color: hashCount > 0 ? "blue" : "inherit",
    };

    return (
      <div>
        <p style={style}>{text.replace(/\*/g, "").replace(/#/g, "")}</p>
      </div>
    );
  };

  const addQuestion = () =>
    setQuestionList([...questionList, { value: "", iconType: "minus" }]);
  const removeQuestion = (index) =>
    setQuestionList(questionList.filter((_, i) => i !== index));

  const addAward = () =>
    setAwardList([...awardList, { value: "", iconType: "minus" }]);
  const removeAward = (index) =>
    setAwardList(awardList.filter((_, i) => i !== index));

  const addExperience = () =>
    setExperienceList([...experienceList, { value: "", iconType: "minus" }]);
  const removeExperience = (index) =>
    setExperienceList(experienceList.filter((_, i) => i !== index));
  const openSearchModal = () => {
    setIsModalOpen(true);
  };
  //추가
  const closeSearchModal = () => {
    setIsModalOpen(false);
    setSearchResults([]);
    setSearchLoading(false);
    setSearchError("");
  };

  const handleSearch = async (value) => {
    setSearchLoading(true);
    setSearchError("");

    try {
      const response = await axiosInstance.get(`/resume-items/load/${value}`);
      if (response.data.status === "Not found") {
        setSearchError("Not found");
      } else if (response.data.status === "Success") {
        setSearchResults(response.data.itemsList);
      }
    } catch (error) {
      setSearchError("Failed to search");
    } finally {
      setSearchLoading(false);
    }
  };

  const handleRowClick = (record) => {
    const { company, items } = record;
    const itemList = items.split("||");
    userInputForm.setFieldsValue({ company });
    setQuestionList(
      itemList.map((item, index) => ({
        value: item,
        iconType: index === 0 ? "plus" : "minus",
      }))
    );
    closeSearchModal();
  };

  const openOccupationSearchModal = () => {
    setIsOccupationModalOpen(true);
  };

  const closeOccupationSearchModal = () => {
    setIsOccupationModalOpen(false);
    setOccupationSearchResults([]);
    setOccupationSearchLoading(false);
    setOccupationSearchError("");
  };

  const handleOccupationSearch = async (value) => {
    setOccupationSearchLoading(true);
    setOccupationSearchError("");

    try {
      const response = await axiosInstance.get(
        `/resume-occupation/load/${value}`
      );
      if (response.data.status === "Not found") {
        setOccupationSearchError("Not found");
      } else if (response.data.status === "Success") {
        setOccupationSearchResults(response.data.occupationList);
      }
    } catch (error) {
      setOccupationSearchError("Failed to search");
    } finally {
      setOccupationSearchLoading(false);
    }
  };

  const handleOccupationRowClick = (record) => {
    const { occupation } = record;
    userInputForm.setFieldsValue({ occupation });
    closeOccupationSearchModal();
  };

  const handleInputChange = (e, index, list, setList) => {
    const newValue = e.target.value;
    if (newValue.includes("\n")) {
      const splitValues = newValue.split("\n");
      const newList = [...list];
      newList[index].value = splitValues[0];
      const additionalItems = splitValues
        .slice(1)
        .map((value) => ({ value, iconType: "minus" }));
      setList([...newList, ...additionalItems]);
    } else {
      const newList = [...list];
      newList[index].value = newValue;
      setList(newList);
    }
  };
  const occupationColumns = [
    {
      title: "Occupation",
      dataIndex: "occupation",
      key: "occupation",
    },
  ];
  const columns = [
    {
      title: "Company",
      dataIndex: "company",
      key: "company",
    },
    {
      title: "Items",
      dataIndex: "items",
      key: "items",
    },
  ];
  return (
    <div style={{ padding: "5% 5%" }}>
      <div className="Wrapper" style={{ padding: "2% 5%", display: "flex" }}>
        <div
          className="userInnerWrapper"
          style={{
            border: "1px solid rgb(220,220,220)",
            boxShadow: "0 0 10px 0 rgb(220, 220, 220)",
            borderRadius: "5px",
            height: "100%",
            width: "55%",
          }}
        >
          <div className="userInputWrapper" style={{ padding: "5% 5%" }}>
            <Tooltip
              title="자기소개서 가이드는 수상경력이나 직무 관련 경험을 정리해 놓으면 자기소개서 항목에 분배해주고 예시를 만들어 드려요! 지원 회사를 검색하면 예시 자기소개서 항목을 로드할 수 있어요!"
              placement="topLeft"
              overlayStyle={{ fontSize: "1rem", maxWidth: "400px" }}
            >
              <InfoCircleOutlined
                style={{
                  fontSize: "15px",
                  top: "10px",
                  bottom: "10px",
                  left: "10px",
                  cursor: "pointer",
                }}
              />
            </Tooltip>
            <Form layout={"vertical"} form={userInputForm} onFinish={onFinish}>
              <Form.Item
                name="status"
                label={<b>신입/경력</b>}
                style={{
                  display: "inline-block",
                  width: "calc(50% - 8px)",
                  marginTop: "15px",
                }}
              >
                <Radio.Group>
                  <Radio value="신입"> 신입 </Radio>
                  <Radio value="경력"> 경력 </Radio>
                </Radio.Group>
              </Form.Item>

              <Form.Item style={{ marginBottom: 0 }}>
                <Form.Item
                  name="company"
                  label={<b>지원 회사</b>}
                  style={{ display: "inline-block", width: "calc(50% - 8px)" }}
                >
                  <Input
                    suffix={
                      <Button
                        icon={<SearchOutlined />}
                        onClick={openSearchModal}
                      />
                    }
                    placeholder="회사 이름"
                    size="large"
                  />
                </Form.Item>
                <Form.Item
                  label={<b>지원 직무</b>}
                  name="occupation"
                  style={{
                    display: "inline-block",
                    width: "calc(50% - 8px)",
                    margin: "0 8px",
                  }}
                >
                  <Input
                    suffix={
                      <Button
                        icon={<SearchOutlined />}
                        onClick={openOccupationSearchModal}
                      />
                    }
                    placeholder="직무 이름"
                    size="large"
                  />
                </Form.Item>
              </Form.Item>

              <div>
                <b>자소서 문항</b>
                {questionList.map((q, index) => (
                  <div
                    key={index}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      marginBottom: "8px",
                    }}
                  >
                    {q.iconType === "plus" ? (
                      <PlusOutlined
                        onClick={addQuestion}
                        style={{ cursor: "pointer" }}
                      />
                    ) : (
                      <MinusOutlined
                        onClick={() => removeQuestion(index)}
                        style={{ cursor: "pointer" }}
                      />
                    )}
                    <Input
                      style={{ marginLeft: "8px" }}
                      value={q.value}
                      onChange={(e) =>
                        handleInputChange(
                          e,
                          index,
                          questionList,
                          setQuestionList
                        )
                      }
                    />
                  </div>
                ))}
              </div>

              <div>
                <b>수상 경력</b>
                {awardList.map((a, index) => (
                  <div
                    key={index}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      marginBottom: "8px",
                    }}
                  >
                    {a.iconType === "plus" ? (
                      <PlusOutlined
                        onClick={addAward}
                        style={{ cursor: "pointer" }}
                      />
                    ) : (
                      <MinusOutlined
                        onClick={() => removeAward(index)}
                        style={{ cursor: "pointer" }}
                      />
                    )}
                    <Input
                      style={{ marginLeft: "8px" }}
                      value={a.value}
                      onChange={(e) =>
                        handleInputChange(e, index, awardList, setAwardList)
                      }
                    />
                  </div>
                ))}
              </div>

              <div>
                <b>직무 관련 경험</b>
                {experienceList.map((e, index) => (
                  <div
                    key={index}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      marginBottom: "8px",
                    }}
                  >
                    {e.iconType === "plus" ? (
                      <PlusOutlined
                        onClick={addExperience}
                        style={{ cursor: "pointer" }}
                      />
                    ) : (
                      <MinusOutlined
                        onClick={() => removeExperience(index)}
                        style={{ cursor: "pointer" }}
                      />
                    )}
                    <Input
                      style={{ marginLeft: "8px" }}
                      value={e.value}
                      onChange={(e) =>
                        handleInputChange(
                          e,
                          index,
                          experienceList,
                          setExperienceList
                        )
                      }
                    />
                  </div>
                ))}
              </div>

              <Form.Item style={{ textAlign: "center", marginTop: "40px" }}>
                <Button
                  onClick={() => {
                    setIsLoading(true);
                  }}
                  style={{ backgroundColor: "#85dad2" }}
                  type="primary"
                  htmlType="submit"
                  size="large"
                >
                  생성하기
                </Button>
              </Form.Item>
            </Form>
          </div>
        </div>
        <div
          className="gptInnerWrapper"
          style={{
            border: "1px solid rgb(220,220,220)",
            boxShadow: "0 0 10px 0 rgb(220, 220, 220)",
            borderRadius: "5px",
            height: "auto",
            width: "50%",
            marginLeft: "4%",
          }}
        >
          <div
            className="gptResultWrapper"
            style={{ padding: "5% 5%", height: "100%" }}
          >
            <div className="gptResult" style={{ height: "100%" }}>
              {generated ? (
                isLoading ? (
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center",
                      height: "100%",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        textAlign: "center",
                      }}
                    >
                      {randomSpinner()}
                    </div>
                  </div>
                ) : (
                  <div>
                    {result.split("\n").map((text, index) => (
                      <div key={index}>{applyStyleToText(text)}</div>
                    ))}
                    <div
                      style={{
                        width: "100%",
                        display: "flex",
                        justifyContent: "flex-end",
                      }}
                    >
                      <Button
                        style={{ backgroundColor: "#85DAD2", color: "white" }}
                        size="large"
                        onClick={() => {
                          setOpenRateModal(true);
                        }}
                      >
                        후기 남기기
                      </Button>
                    </div>
                  </div>
                )
              ) : (
                <div
                  style={{
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    fontWeight: "bold",
                    fontSize: "1.1rem",
                  }}
                >
                  자기소개서 가이드가 이 곳에 출력돼요!
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      <Modal
        title="Search Company"
        visible={isModalOpen}
        onCancel={closeSearchModal}
        footer={null}
        width={600}
      >
        <Input.Search
          placeholder="Search company"
          enterButton
          onSearch={handleSearch}
        />
        {searchLoading ? (
          <div style={{ textAlign: "center", marginTop: "20px" }}>
            <PuffLoader color="#36d7b7" size={20} />
          </div>
        ) : searchError ? (
          <div style={{ textAlign: "center", marginTop: "20px", color: "red" }}>
            {searchError}
          </div>
        ) : (
          <Table
            columns={columns}
            dataSource={searchResults}
            rowKey="company"
            onRow={(record) => ({
              onClick: () => handleRowClick(record),
            })}
            pagination={false}
            style={{ marginTop: "20px" }}
          />
        )}
      </Modal>
      <Modal
        title="Search Occupation"
        visible={isOccupationModalOpen}
        onCancel={closeOccupationSearchModal}
        footer={null}
        width={600}
      >
        <Input.Search
          placeholder="Search occupation"
          enterButton
          onSearch={handleOccupationSearch}
        />
        {occupationSearchLoading ? (
          <div style={{ textAlign: "center", marginTop: "20px" }}>
            <PuffLoader color="#36d7b7" size={20} />
          </div>
        ) : occupationSearchError ? (
          <div style={{ textAlign: "center", marginTop: "20px", color: "red" }}>
            {occupationSearchError}
          </div>
        ) : (
          <Table
            columns={occupationColumns}
            dataSource={occupationSearchResults}
            rowKey="occupation"
            onRow={(record) => ({
              onClick: () => handleOccupationRowClick(record),
            })}
            pagination={false}
            style={{ marginTop: "20px" }}
          />
        )}
      </Modal>
      <Modal
        title={
          <div style={{ textAlign: "center" }}>
            해당 글에 대한 후기를 남겨주세요!
          </div>
        }
        centered
        open={openRateModal}
        onCancel={() => {
          setOpenRateModal(false);
        }}
        footer={[
          <div
            key={"onOk"}
            style={{ width: "100%", display: "flex", justifyContent: "center" }}
          >
            <Button
              style={{ backgroundColor: "#85DAD2", color: "white" }}
              size="large"
              onClick={() => {
                let accessToken = localStorage.getItem("access") ?? "";
                let DecodedToken: DecodedToken = jwtDecode(accessToken);
                let res = axiosInstance

                  .post("/board/rating", {
                    rating: rating,
                    unum: DecodedToken.uNum,
                  })
                  .then((res) => {
                    setOpenRateModal(false);

                    Swal.fire({
                      icon: "success",
                      title: "후기가 등록되었습니다!",
                      text: "감사합니다 :)",
                    });
                  });
              }}
            >
              평가하기!
            </Button>
          </div>,
        ]}
      >
        <div className="star-rating">
          {[...Array(5)].map((_, index) => {
            const value = (index + 1) * 2;
            const halfValue = value - 1;
            return (
              <span
                key={`star${index}`}
                className="star"
                onClick={() => {
                  handleClick(value / 2);
                }}
                onMouseOver={() => handleMouseOver(value)}
                onMouseOut={handleMouseOut}
              >
                {hover >= value || rating >= value / 2 ? (
                  <StarFilled
                    style={{ color: hover >= value ? "#ffc107" : "#ffa500" }}
                  />
                ) : hover >= halfValue || rating >= halfValue / 2 ? (
                  <StarFilled
                    style={{
                      color: hover >= halfValue ? "#ffc107" : "#ffa500",
                    }}
                    className="half"
                  />
                ) : (
                  <StarOutlined style={{ color: "#ddd" }} />
                )}
              </span>
            );
          })}
        </div>
        <div>
          <TextArea
            style={{ fontSize: "20px" }}
            rows={5}
            placeholder="후기를 남겨주세요"
          ></TextArea>
        </div>
      </Modal>
    </div>
  );
};

export default ResumeGuide;
