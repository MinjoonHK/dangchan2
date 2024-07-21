import { useForm } from "antd/es/form/Form";
import Button from "antd/es/button";
import DatePicker from "antd/es/date-picker";
import Form from "antd/es/form";
import Input from "antd/es/input";
import Radio from "antd/es/radio";
import Select from "antd/es/select";
import Modal from "antd/es/modal";
import { Link, useNavigate } from "react-router-dom";
import CustomFooter from "../../components/footer";
import { ChangeEvent, useState, useEffect } from "react";
import Swal from "sweetalert2";
import axios from "axios";
import Timer from "./timer";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "@/store/store";
import { start } from "@/store/features/timer/timerSlice";
import React from "react";
import "./signUp.scss";
import { ServiceAgreement } from "./serviceAgreement";
import { PrivateInfoAgreement } from "./privateInfoAgreement";
import { Checkbox } from "antd";

const SignUp = () => {
  let [submitForm] = useForm();
  const navigate = useNavigate();
  const [userAge, setUserAge] = useState(0);
  const [selectedStatus, setSelectedStatus] = useState(-1);
  const [emailSend, setEmailSend] = useState(false);
  const [verified, setVerified] = useState(false);
  const [isModalVisible, setIsModalVisible] = useState(true);
  const [serviceAgreeCheckbox, setServiceAgreeCheckbox] = useState(false);
  const [privateInfoAgreeCheckbox, setPrivateInfoAgreeCheckbox] =
    useState(false);
  const [allAgreeCheckbox, setAllAgreeCheckbox] = useState(false);
  const timer = useSelector((state: RootState) => state.timer.value);
  const dispatch = useDispatch();

  const onSubmitForm = async ({
    email,
    password,
    gender,
    userBirthDate,
    username,
    age,
    company,
    wish,
    occupation,
    status,
  }) => {
    if (!verified) {
      Swal.fire({
        icon: "error",
        title: "이메일 인증을 완료해주세요!",
        showConfirmButton: true,
        timer: 2000,
      });
    } else {
      let birthDate = `${userBirthDate.get("year")}-${
        userBirthDate.get("month") + 1
      }-${userBirthDate.get("date")}`;
      let role = "user";
      let name = "user";
      try {
        let response = await axios.post(
          "/signup",
          {
            email,
            password,
            company,
            occupation,
            username,
            birthDate,
            role,
            name,
            gender,
            age,
            wish,
            status,
          },
          {
            withCredentials: true,
          }
        );
        if (response) {
          Swal.fire({
            icon: "success",
            title: "회원가입이 완료되었습니다!",
            showConfirmButton: true,
            timer: 1500,
          });
          navigate("/");
        } else {
          console.log(response);
        }
      } catch (err) {
        console.log(err);
      }
    }
  };

  const removeKorean = (event: ChangeEvent<HTMLInputElement>) => {
    event.target.value = event.target.value.replace(/[ㄱ-ㅎㅏ-ㅣ가-힣]/g, "");
  };

  const handleOk = () => {
    setIsModalVisible(false);
  };

  const handleAgree = () => {
    if (serviceAgreeCheckbox && privateInfoAgreeCheckbox) {
      handleOk();
    } else {
      Swal.fire({
        icon: "error",
        title: "동의 필요",
        text: "모든 약관에 동의하셔야 합니다.",
      });
    }
  };

  const handleAllAgreeChange = () => {
    const newValue = !(serviceAgreeCheckbox && privateInfoAgreeCheckbox);
    setServiceAgreeCheckbox(newValue);
    setPrivateInfoAgreeCheckbox(newValue);
    setAllAgreeCheckbox(newValue);
  };

  const handleIndividualChange = (serviceChecked, privateChecked) => {
    setServiceAgreeCheckbox(serviceChecked);
    setPrivateInfoAgreeCheckbox(privateChecked);
    setAllAgreeCheckbox(serviceChecked && privateChecked);
  };

  useEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      if (isModalVisible) {
        event.preventDefault();
        event.returnValue = "";
      }
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [isModalVisible]);

  return (
    <div className="signUpOuterContainer">
      <div className="signUpTitle">Resume Editor</div>
      <div className="signUpContainer">
        <div className="signUpFormWrapper">
          <div className="signUpFormInnerWrapper">
            <div className="signUpInnerContent">
              <div className="signUpWelcomeMessage">가입을 환영합니다!</div>
              <div className="signUpSubMessage">
                <p>고객님에 대해 더 자세히 알아가고 싶어요!</p>
              </div>
              <div>
                <span>이미 계정이 있으신가요?</span>
                <Link className="backToLogin" to="/auth/login">
                  로그인
                </Link>
              </div>
              <div className="signUpBtnWrapper">
                <div className="signUpForm">
                  <Form
                    form={submitForm}
                    onFinish={onSubmitForm}
                    layout="vertical"
                    initialValues={{ gender: "성별을 선택해 주세요" }}
                    fields={[{ name: ["age"], value: userAge }]}
                  >
                    <Form.Item
                      label={<b>이메일</b>}
                      name="email"
                      style={{ marginBottom: "3%" }}
                      rules={[
                        () => ({
                          validator(_, value) {
                            let emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                            let emailCheck = emailRegex.test(value);
                            if (!value) {
                              return Promise.reject(
                                new Error("이메일을 입력해 주세요!")
                              );
                            } else if (!emailCheck) {
                              return Promise.reject(
                                new Error("유효하지 않은 이메일 형식 입니다!")
                              );
                            }
                            return Promise.resolve();
                          },
                        }),
                      ]}
                    >
                      <Input size="large" placeholder="이메일 입력" />
                    </Form.Item>
                    {emailSend ? (
                      <Form.Item style={{ marginBottom: "0" }}>
                        <Form.Item
                          name="authCode"
                          style={{
                            display: "inline-block",
                            width: "calc(77% - 8px)",
                          }}
                        >
                          <Input placeholder="인증번호 입력" size="large" />
                        </Form.Item>

                        <Form.Item
                          style={{
                            display: "inline-block",
                            width: "calc(23% - 8px)",
                            margin: "0 8px",
                          }}
                        >
                          <Button size="large">
                            <Timer />
                          </Button>
                        </Form.Item>
                        <Form.Item>
                          <Button
                            onClick={() => {
                              if (
                                submitForm.getFieldValue("authCode") ==
                                undefined
                              ) {
                                Swal.fire({
                                  icon: "error",
                                  title: "인증번호를 입력해주세요!",
                                  showConfirmButton: true,
                                  timer: 2000,
                                });
                              }
                              let checkCode = axios
                                .post("/signup/auth-check", {
                                  email: submitForm.getFieldValue("email"),
                                  authCode:
                                    submitForm.getFieldValue("authCode"),
                                })
                                .then((res) => {
                                  if (res.status == 200) {
                                    setVerified(true);
                                  }
                                })
                                .catch((err) => {
                                  Swal.fire({
                                    icon: "error",
                                    title: "인증번호가 일치하지 않습니다!",
                                    showConfirmButton: true,
                                    timer: 2000,
                                  });
                                });
                            }}
                            style={{ width: "100%" }}
                            size="large"
                          >
                            {verified ? "확인 되었습니다!" : "인증번호 확인"}
                          </Button>
                        </Form.Item>
                      </Form.Item>
                    ) : (
                      <Form.Item>
                        <Button
                          onClick={() => {
                            if (
                              submitForm.getFieldValue("email") == undefined
                            ) {
                              Swal.fire({
                                icon: "error",
                                title: "이메일을 입력해주세요!",
                                showConfirmButton: true,
                                timer: 2000,
                              });
                            } else {
                              let email = submitForm.getFieldValue("email");
                              let response = axios
                                .post(
                                  "/signup/auth-code",
                                  {
                                    email,
                                  },
                                  {
                                    headers: {
                                      "Content-Type": "application/json",
                                    },
                                  }
                                )
                                .then((res) => {
                                  setEmailSend(true);
                                  dispatch(start());
                                })
                                .catch((err) => {
                                  if (err.response.status === 500) {
                                    Swal.fire({
                                      icon: "error",
                                      title: "이미 가입된 이메일 입니다!",
                                      showConfirmButton: true,
                                      timer: 2000,
                                    });
                                  } else if (err.response.status === 400) {
                                    let deleted =
                                      err.response.data.response.deleted;
                                    let available =
                                      err.response.data.response.available;
                                    Swal.fire({
                                      icon: "error",
                                      title: "탈퇴한 계정입니다",
                                      html: `해당 이메일은 ${deleted}에 탈퇴하였습니다.<br/> ${available}에 재가입이 가능합니다.`,
                                      showConfirmButton: true,
                                      timer: 2000,
                                    });
                                  }
                                });
                            }
                          }}
                          style={{ width: "100%" }}
                          size="large"
                        >
                          이메일 인증하기
                        </Button>
                      </Form.Item>
                    )}
                    <Form.Item
                      label={<b>비밀번호</b>}
                      style={{ marginBottom: "3%" }}
                      name="password"
                      rules={[
                        ({ getFieldValue }) => ({
                          validator(_, value) {
                            let pattern = /[!@#$%^&*()]/;
                            let patternCheck = pattern.test(value);
                            if (!value) {
                              return Promise.reject(
                                new Error("비밀 번호를 입력해 주세요!")
                              );
                            } else if (!patternCheck) {
                              return Promise.reject(
                                new Error("특수문자를 포함해 주세요!")
                              );
                            } else if (value.length < 8) {
                              return Promise.reject(
                                new Error("비밀번호는 8자 이상이어야 합니다!")
                              );
                            }
                            return Promise.resolve();
                          },
                        }),
                      ]}
                    >
                      <Input
                        size="large"
                        type="password"
                        placeholder="비밀번호 입력"
                      />
                    </Form.Item>
                    <Form.Item
                      label={<b>비밀번호 확인</b>}
                      style={{ marginBottom: "3%" }}
                      name="passwordConfirm"
                      dependencies={["password"]}
                      hasFeedback
                      rules={[
                        ({ getFieldValue }) => ({
                          validator(_, value) {
                            if (!value || getFieldValue("password") === value) {
                              return Promise.resolve();
                            }
                            return Promise.reject(
                              new Error("비밀 번호가 일치하지 않습니다!")
                            );
                          },
                        }),
                      ]}
                    >
                      <Input
                        size="large"
                        type="password"
                        placeholder="비밀번호 확인"
                      />
                    </Form.Item>
                    <Form.Item style={{ marginBottom: "0" }}>
                      <Form.Item
                        label={<b>아이디</b>}
                        name="username"
                        style={{
                          display: "inline-block",
                          width: "calc(50% - 8px)",
                        }}
                      >
                        <Input
                          size="large"
                          placeholder="아이디"
                          onInput={removeKorean}
                        />
                      </Form.Item>
                      <Form.Item
                        label={<b>성별</b>}
                        name="gender"
                        style={{
                          margin: "0 8px",

                          display: "inline-block",
                          width: "calc(50% - 8px)",
                        }}
                      >
                        <Select size="large">
                          <Select.Option value="M" label="남성">
                            남성
                          </Select.Option>
                          <Select.Option value="F" label="여성">
                            여성
                          </Select.Option>
                        </Select>
                      </Form.Item>
                    </Form.Item>
                    <Form.Item style={{ marginBottom: "0" }}>
                      <Form.Item
                        label={<b>생년월일</b>}
                        name="userBirthDate"
                        style={{
                          display: "inline-block",
                          width: "calc(50% - 8px)",
                        }}
                      >
                        <DatePicker
                          size="large"
                          onChange={(date) => {
                            let age = new Date().getFullYear() - date.year();
                            setUserAge(age);
                            setTimeout(() => {});
                          }}
                          placeholder="2024-01-01"
                        />
                      </Form.Item>
                      <Form.Item
                        label={<b>나이</b>}
                        name="age"
                        style={{
                          margin: "0 8px",
                          display: "inline-block",
                          width: "calc(50% - 8px)",
                        }}
                      >
                        <Input size="large" disabled />
                      </Form.Item>
                    </Form.Item>
                    <Form.Item
                      label={<b>현재 어떤 상태 이신가요?</b>}
                      name="status"
                    >
                      <Radio.Group
                        size="large"
                        onChange={(e) => setSelectedStatus(e.target.value)}
                      >
                        <Radio value={1}>이직중</Radio>
                        <Radio value={2}>취준중</Radio>
                      </Radio.Group>
                    </Form.Item>
                    <Form.Item
                      label={<b>회사</b>}
                      name="company"
                      style={{ marginBottom: "3%" }}
                    >
                      {selectedStatus == 1 ? (
                        <Input size="large" placeholder="회사" />
                      ) : (
                        <Input size="large" placeholder="회사" disabled />
                      )}
                    </Form.Item>
                    <Form.Item
                      label={<b>직종</b>}
                      name="occupation"
                      style={{ marginBottom: "3%" }}
                    >
                      {selectedStatus == 1 ? (
                        <Select size="large" placeholder="직종을 선택해 주세요">
                          <Select.Option value="개발자">개발자</Select.Option>
                          <Select.Option value="전문직">전문직</Select.Option>
                          <Select.Option value="생산직">생산직</Select.Option>
                        </Select>
                      ) : (
                        <Select
                          size="large"
                          placeholder="직종을 선택해 주세요"
                          disabled
                        ></Select>
                      )}
                    </Form.Item>
                    <Form.Item label={<b>이직/취업 희망분야</b>} name="wish">
                      <Input size="large" placeholder="분야" />
                    </Form.Item>

                    <Form.Item>
                      <Button
                        htmlType="submit"
                        size="large"
                        style={{
                          width: "100%",
                          backgroundColor: "#85dad2",
                          color: "white",
                        }}
                        disabled={isModalVisible}
                      >
                        회원가입
                      </Button>
                    </Form.Item>
                  </Form>
                </div>
              </div>
            </div>
          </div>
        </div>
        <Modal
          width={600}
          title={
            <div
              style={{
                fontSize: "20px",
                textAlign: "center",
                fontFamily: "Prtendard-regular",
              }}
            >
              <b>Reditor 약관동의</b>
            </div>
          }
          open={isModalVisible}
          onOk={handleAgree}
          footer={[
            <div
              key={"footer1"}
              style={{ display: "flex", justifyContent: "center" }}
            >
              <Button
                size="large"
                style={{
                  width: "70%",
                  backgroundColor: "#85dad2",
                  color: "white",
                }}
                key="agree"
                type="primary"
                onClick={handleAgree}
              >
                동의합니다
              </Button>
            </div>,
          ]}
          closable={false}
          maskClosable={false}
        >
          <p>
            <Checkbox
              checked={serviceAgreeCheckbox}
              onChange={(e) =>
                handleIndividualChange(
                  e.target.checked,
                  privateInfoAgreeCheckbox
                )
              }
              style={{
                fontFamily: "Pretendard-SemiBold",
                textDecoration: "underline",
              }}
            >
              서비스 이용약관 동의 (필수)
            </Checkbox>
          </p>
          <div
            style={{
              borderRadius: "5px",
              height: "150px",
              overflow: "auto",
              border: "1px solid #DDDFE1",
            }}
          >
            <ServiceAgreement />
          </div>
          <p>
            <Checkbox
              checked={privateInfoAgreeCheckbox}
              onChange={(e) =>
                handleIndividualChange(serviceAgreeCheckbox, e.target.checked)
              }
              style={{
                fontFamily: "Pretendard-SemiBold",
                textDecoration: "underline",
              }}
            >
              개인정보 수집 및 이용 동의 (필수)
            </Checkbox>
          </p>
          <div
            style={{
              borderRadius: "5px",
              height: "150px",
              overflow: "auto",
              border: "1px solid #DDDFE1",
            }}
          >
            <PrivateInfoAgreement />
          </div>
          <p style={{ marginBottom: "7%" }}>
            <Checkbox
              checked={allAgreeCheckbox}
              onChange={handleAllAgreeChange}
              style={{
                fontFamily: "Pretendard-SemiBold",
                textDecoration: "underline",
              }}
            >
              개인정보처리 및 서비스 이용약관에 모두 동의합니다.
            </Checkbox>
          </p>
        </Modal>
      </div>
      <CustomFooter />
    </div>
  );
};

export default SignUp;
