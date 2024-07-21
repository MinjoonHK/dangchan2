import Button from "antd/es/button";
import Form from "antd/es/form";
import Input from "antd/es/input";
import notification from "antd/es/notification";
import axios from "axios";
import { motion, useAnimation } from "framer-motion";
import React, { useEffect } from "react";
import { useDispatch } from "react-redux";
import { Link, useNavigate } from "react-router-dom";
import { login } from "@/store/features/user/userSlice";
import { DecodedToken } from "@/types/globalTypes";
import { jwtDecode } from "jwt-decode";
import { setToken } from "@/store/features/token/tokenSlice";
import Swal from "sweetalert2";

const LoginLanding = () => {
  const controls1 = useAnimation();
  const controls2 = useAnimation();
  const controls3 = useAnimation();
  const controls4 = useAnimation();
  const controls5 = useAnimation();

  useEffect(() => {
    const sequence = async () => {
      await controls1.start({ opacity: 1, scale: 1 });
      await controls2.start({ opacity: 1, scale: 1 });
      await controls3.start({ opacity: 1, scale: 1 });
      await controls4.start({ opacity: 1, scale: 1 });
      await controls5.start({ opacity: 1, scale: 1 });
    };
    sequence();
  }, [controls1, controls2, controls3, controls4, controls5]);

  const [notify, contextHolder] = notification.useNotification();
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const callNotification = () => {
    notify.error({
      message: "로그인 실패",
      description: "아이디와 비밀번호를 확인해주세요",
      placement: "topRight",
      duration: 5,
    });
  };

  const tryLogin = async ({ username, password }) => {
    try {
      let response = await axios
        .post("/login", {
          username: username,
          password: password,
        })
        .then((res) => {
          if (res.status == 200) {
            let accessToken = res.headers["access"];
            let refreshToken = res.headers["refresh"];
            localStorage.setItem("access", accessToken);
            localStorage.setItem("refresh", refreshToken);
            let DecodedToken: DecodedToken = jwtDecode(accessToken);
            dispatch(login());
            dispatch(setToken(DecodedToken));
            if (DecodedToken.role == "ROLE_ADMIN") {
              navigate("/admin/main");
            } else {
              navigate("/main/resume");
            }
          }
        });
    } catch (err) {
      if (err.response.status === 401) {
        console.log(
          err.response.data.message.split(" ")[0],
          err.response.data.message.split("blacklisted").join("")
        );
        if (err.response.data.message.split(" ")[0] === "blacklisted") {
          const blackListedDate = err.response.data.message
            .split("blacklisted")
            .join("")
            .slice(0, 11);
          console.log(blackListedDate);
          Swal.fire({
            icon: "error",
            title: "블랙리스트 계정안내",
            text: `해당 계정은 ${blackListedDate}까지 블랙리스트로 지정되었습니다. 관리자에게 문의해주세요.`,
          });
        } else {
          callNotification();
        }
      }
    }
  };

  return (
    <div>
      {contextHolder}
      <div className="wrapper" style={{ display: "flex" }}>
        <div
          className="leftWrapper"
          style={{
            backgroundColor: "#85dad2",
            width: "50%",
            height: "100vh",
          }}
        >
          <div
            style={{
              justifyContent: "center",
              display: "flex",
              flexDirection: "column",
              color: "white",
              paddingTop: "5%",
              paddingBottom: "15%",
              paddingRight: "15%",
              paddingLeft: "20%",
            }}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.5 }}
              animate={controls1}
              transition={{ duration: 0.5 }}
            >
              <p style={{ fontSize: "3rem", fontWeight: "bold" }}>
                당신만을 위한 자소서 AI 자소서 컨설턴트!
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, scale: 0.5 }}
              animate={controls2}
              transition={{ duration: 0.5 }}
            >
              <p style={{ fontSize: "1.2rem" }}>
                "당신의 경력과 능력을 효과적으로 표현하는 자소서
                <br />
                Reditor가 도와드리겠습니다!"
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, scale: 0.5 }}
              animate={controls3}
              transition={{ duration: 0.5 }}
            >
              <p style={{ fontSize: "1.2rem" }}>
                "수많은 자기소개서 중에서도 돋보일 수 있도록, 당신의 강점을
                최대한 부각시키는 자소서를 작성해드립니다."
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, scale: 0.5 }}
              animate={controls4}
              transition={{ duration: 0.5 }}
            >
              <p style={{ fontSize: "1.2rem" }}>
                "AI의 정교한 분석을 통해 맞춤형 자소서를 작성하여,
                <br /> 취업 경쟁력을 높이세요!"
              </p>
            </motion.div>
          </div>
        </div>
        <div
          className="rightWrapper"
          style={{
            height: "100vh",
            width: "50%",
            backgroundColor: "#85dad2",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              width: "100%",
              display: "flex",
              justifyContent: "center",

              alignItems: "center",
            }}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.5 }}
              animate={controls5}
              transition={{ duration: 0.5 }}
              style={{
                color: "white",
                width: "55%",
                textAlign: "center",
                fontSize: "36px",
                border: "3px solid white",
                borderRadius: "20px",
              }}
            >
              Reditor
            </motion.div>
          </div>
          <div
            className="signUpWrpper"
            style={{
              width: "40vw",
              height: "60%",
            }}
          >
            <div
              className="signUpInnerWrapper"
              style={{
                textAlign: "center",
                display: "flex",
                justifyContent: "center",
              }}
            >
              <div
                className="formWrapper"
                style={{
                  marginTop: "10%",
                  width: "60%",
                  backgroundColor: "white",
                  padding: "5%",
                  border: "1px solid rgb(224,224,224)",
                  borderRadius: "20px",
                }}
              >
                <Form onFinish={tryLogin}>
                  <Form.Item name="username" style={{ marginBottom: "2%" }}>
                    <Input
                      style={{ width: "100%" }}
                      size="large"
                      placeholder="아이디 입력"
                    />
                  </Form.Item>
                  <Form.Item name="password">
                    <Input
                      type="password"
                      style={{ width: "100%" }}
                      size="large"
                      placeholder="비밀번호 입력"
                    />
                  </Form.Item>
                  <Form.Item>
                    <Button
                      htmlType="submit"
                      style={{
                        width: "100%",
                        color: "white",
                        backgroundColor: "#85dad2",
                        border: "1px solid white",
                        fontWeight: "bold",
                        fontSize: "16px",
                        height: "50px",
                      }}
                    >
                      로그인
                    </Button>
                  </Form.Item>
                </Form>
                {/* <Divider
                  orientation="center"
                  style={{ color: "rgb(200,200,200)" }}
                >
                  또는 */}
                {/* </Divider> */}
                {/* <Button
                  style={{
                    width: "100%",
                    backgroundColor: "#00C73C",
                    color: "white",
                    display: "flex",
                    textAlign: "center",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: "bold",
                    fontSize: "16px",
                    height: "50px",
                  }}
                >
                  <Avatar size={30} src={"/img/naver_icon.png"} />
                  네이버 로그인
                </Button> */}
                <div
                  style={{
                    display: "flex",
                    justifyContent: "right",
                    textDecoration: "underline",
                    textDecorationColor: "#85dad2",
                    marginTop: "10px",
                  }}
                >
                  <Link to="/auth/signup">
                    <span
                      style={{
                        color: "#85dad2",
                      }}
                    >
                      회원가입
                    </span>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* <LandingComment /> */}
      {/* <Box /> */}
    </div>
  );
};

export default LoginLanding;
