import { AxiosInstance } from "axios";
import { JwtPayload, jwtDecode } from "jwt-decode";

export const tokenRefresh = async (instance: AxiosInstance) => {
  const refreshToken = localStorage.getItem("refresh"); // 리프레시 토큰을 가져오기
  if (!refreshToken) {
    // 리프레시 토큰이 없는 경우
    return false;
  }
  const data = await instance
    .post("/reissue", {
      headers: {
        "Content-Type": "application/json",
        refresh: refreshToken,
      },
    })
    .then((res) => {
      const newAccessToken = res.headers["access"];
      localStorage.setItem("access", newAccessToken);
      return true;
    })
    .catch((err) => {
      console.log(err);
      return false;
    });
}; // tokenRefresh() - 토큰을 갱신해주는 함수

export const isTokenExpired = () => {
  // 토큰 만료 검사
  const accessToken = localStorage.getItem("access");
  if (!accessToken) {
    // 토큰이 없는경우
    return true;
  }
  const decodedToken = jwtDecode<JwtPayload>(accessToken);
  const currentTime = Date.now() / 1000;
  if (decodedToken.exp !== undefined && decodedToken.exp < currentTime) {
    // 토큰이 만료된 경우
    return true;
  }
  return false;
};
