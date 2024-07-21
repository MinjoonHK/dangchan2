import { Link, useNavigate } from "react-router-dom";
import "./navbar.css";
import React from "react";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "@/store/store";
import Button from "antd/es/button";
import { useEffect } from "react";
import { login } from "@/store/features/user/userSlice";
import Swal from "sweetalert2";
import { jwtDecode } from "jwt-decode";
import { DecodedToken } from "@/store/features/token/tokenSlice";
import axiosInstance from "@/api/api";

const Navbar = () => {
  const user = useSelector((state: RootState) => state.user.value);
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const accessToken = localStorage.getItem("access") ?? "";
  const decodedToken: DecodedToken = jwtDecode(accessToken);
  const username = decodedToken.username;

  useEffect(() => {
    let accessToken = localStorage.getItem("access");
    if (!accessToken) {
      Swal.fire({
        title: "로그인을 먼저 해주세요",
        icon: "warning",
        confirmButtonColor: "#1890ff",
      }).then((res) => {
        navigate("/");
      });
    } else if (accessToken) {
      dispatch(login());
    }
  }, []);
  return (
    <div>
      <nav className="navbar">
        <div className="navbar_logo">
          {accessToken == "" ? (
            <Link to="/">Reditor</Link>
          ) : (
            <Link to="/main/resume">Reditor</Link>
          )}
        </div>

        <ul className="navbar_menu">
          <li>
            <Link to="/main/resumeguide">자소서 가이드</Link>
          </li>
          <li>
            <Link to="/main/resume">자소서 첨삭</Link>
          </li>
          <li>
            <Link to="/main/recommendation">맞춤 일자리</Link>
          </li>
          <li>
            <Link to="/main/resumelist">자소서 목록</Link>
          </li>
          <li>
            <Link to="/main/mypage">마이페이지</Link>
          </li>
          <li>
            <Link to="/main/faq">FAQ</Link>
          </li>
        </ul>
        {user == true ? (
          <ul className="navbar_links" style={{ alignItems: "center" }}>
            <li>{username} 님</li>
            <li>
              <Button
                style={{
                  backgroundColor: "transparent",
                  border: "1px solid white",
                  color: "white",
                  fontWeight: "bold",
                }}
                onClick={() => {
                  let refreshToken = localStorage.getItem("refresh");
                  let res = axiosInstance
                    .post(
                      "/logout",
                      {},
                      {
                        headers: {
                          refresh: refreshToken,
                        },
                      }
                    )
                    .then((res) => {
                      if (res.status == 200) {
                        localStorage.removeItem("access");
                        localStorage.removeItem("refresh");
                        navigate("/");
                      }
                    });
                }}
              >
                로그아웃
              </Button>
            </li>
          </ul>
        ) : (
          <ul className="navbar_links">
            <li>
              <Link to="/">로그인</Link>
            </li>
            <li>
              <Link to="/auth/signup">회원가입</Link>
            </li>
          </ul>
        )}
      </nav>
    </div>
  );
};

export default Navbar;
