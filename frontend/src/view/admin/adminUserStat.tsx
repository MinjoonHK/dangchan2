import axiosInstance from "@/api/api";
import { useEffect, useState } from "react";
import DougnutGraph from "./dougnutGraph";
import Divider from "antd/es/divider";
import DatePicker from "antd/es/date-picker";
import Button from "antd/es/button";
import Dropdown from "antd/es/dropdown";
import type { MenuProps } from "antd";
import LineGraph from "./lineGraph";
import React from "react";
import dayjs from "dayjs";
import customParseFormat from "dayjs/plugin/customParseFormat";

dayjs.extend(customParseFormat);

const { RangePicker } = DatePicker;
const dateFormat = "YYYY-MM-DD";
const today = dayjs();

const AdminUserStat = () => {
  const [userCount, setUserCount] = useState(0);
  const [userProCount, setUserProCount] = useState(0);
  const [totalVisit, setTotalVisit] = useState(0);
  const [todayVisit, setTodayVisit] = useState(0);
  const [userGender, setUserGender] = useState({});
  const [employRate, setEmployRate] = useState({});
  const [ageRange, setAgeRange] = useState({});
  const [companyRanking, setCompanyRanking] = useState({});
  const [occupationRanking, setOccupationRanking] = useState({});
  const [wishRanking, setWishRanking] = useState({});
  const [trafficData, setTrafficData] = useState({});
  const [signupData, setSignupData] = useState({});
  const [selectedType, setSelectedType] = useState("타입");

  const fetchAdminStatData = async (group: string) => {
    let res = await axiosInstance
      .get(`admin/statistics/rank/${group}`)
      .then((res) => {
        switch (group) {
          case "occupation":
            setOccupationRanking(res.data.response.ranking_user);
            break;
          case "company":
            setCompanyRanking(res.data.response.ranking_user);
            break;
          case "wish":
            setWishRanking(res.data.response.ranking_user);
            break;
        }
      })
      .catch((err) => {
        console.log("박스데이터를 가져오는데 오류가 발생했습니다", err);
      });
  };
  const fetchUserStatData = async (group: string) => {
    let res = await axiosInstance
      .get(`/statistics/user/${group}`)
      .then((res) => {
        switch (group) {
          case "count":
            setUserCount(res.data.total_count);
            break;
          case "gender":
            setUserGender(res.data);
            break;
          case "age":
            setAgeRange(res.data);
            break;
          case "status":
            setEmployRate(res.data);
            break;
          case "pro":
            setUserProCount(res.data.pro);
            break;
          case "total-visit":
            setTotalVisit(res.data.visit_total);
            break;
          case "today-visit":
            setTodayVisit(res.data.visit_today);
            break;
        }
      })
      .catch((err) => {
        console.log("도넛데이터를 가져오는데 오류가 발생했습니다", err);
      });
  };

  const fetchAccumulatedData = async (group: string) => {
    try {
      const res = await axiosInstance
        .get(`/statistics/user/${group}`)
        .then((res) => {
          switch (group) {
            case "access":
              setTrafficData(res.data.traffic_date);
              break;
            case "signup":
              setSignupData(res.data.signup_date);
              break;
          }
        });
    } catch (err) {
      console.log("집계데이터를 불러오는데에 실패했습니다!", err);
    }
  };

  const fetchData = async () => {
    await Promise.all([
      fetchUserStatData("count"),
      fetchUserStatData("pro"),
      fetchUserStatData("total-visit"),
      fetchUserStatData("today-visit"),
      fetchUserStatData("gender"),
      fetchUserStatData("age"),
      fetchUserStatData("status"),
      fetchAdminStatData("occupation"),
      fetchAdminStatData("company"),
      fetchAdminStatData("wish"),
      fetchAccumulatedData("access"),
      fetchAccumulatedData("signup"),
    ]);
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleChangeDebut = (range) => {
    if (selectedType == "월별") {
      const valueOfInput = range.format();
      const res = axiosInstance
        .get("/statistics/user/access", {
          params: {
            monthly: valueOfInput.slice(0, 10),
          },
        })
        .then((res) => {
          console.log(res);
          setTrafficData(res.data.traffic_date);
        });
      const res2 = axiosInstance
        .get("/statistics/user/signup", {
          params: {
            monthly: valueOfInput.slice(0, 10),
          },
        })
        .then((res) => {
          console.log(res);
          setSignupData(res.data.signup_date);
        });
    } else {
      const valueOfInput1 = range[0].format();
      const valueOfInput2 = range[1].format();
      const res = axiosInstance
        .get("/statistics/user/access", {
          params: {
            startDate: valueOfInput1.slice(0, 10),
            endDate: valueOfInput2.slice(0, 10),
          },
        })
        .then((res) => {
          console.log(res);
          setTrafficData(res.data.traffic_date);
        });
      const res2 = axiosInstance
        .get("/statistics/user/signup", {
          params: {
            startDate: valueOfInput1.slice(0, 10),
            endDate: valueOfInput2.slice(0, 10),
          },
        })
        .then((res) => {
          setSignupData(res.data.signup_date);
        });
    }
  };

  const doughnutData = [
    {
      dataObject: userGender,
      labelName: "유저수",
      graphTitle: "성별 비율",
    },
    {
      dataObject: employRate,
      labelName: "비율",
      graphTitle: "취업 비율",
    },
    {
      dataObject: ageRange,
      labelName: "나잇대",
      graphTitle: "나이 비율",
    },
    {
      dataObject: occupationRanking,
      labelName: "유저수",
      graphTitle: "TOP5 직업랭킹",
    },
    {
      dataObject: companyRanking,
      labelName: "유저수",
      graphTitle: "TOP5 회사랭킹",
    },
    {
      dataObject: wishRanking,
      labelName: "유저수",
      graphTitle: "TOP5 희망직업랭킹",
    },
  ];

  const squareData = [
    {
      dataObject: userCount,
      title: "총 유저수",
    },
    {
      dataObject: userProCount,
      title: "프로 유저수",
    },
    {
      dataObject: totalVisit,
      title: "총 방문자수",
    },
    {
      dataObject: todayVisit,
      title: "오늘 방문자수",
    },
  ];

  const lineData = [
    {
      dataObject: trafficData,
      labelName: "방문자수",
      graphTitle: "주간 방문자수",
    },
    {
      dataObject: signupData,
      labelName: "가입자수",
      graphTitle: "주간 가입자수",
    },
  ];

  const items: MenuProps["items"] = [
    {
      key: "1",
      label: (
        <div
          style={{ textAlign: "center" }}
          onClick={() => setSelectedType("일별")}
        >
          일별
        </div>
      ),
    },
    {
      key: "2",
      label: (
        <div
          style={{ textAlign: "center" }}
          onClick={() => setSelectedType("월별")}
        >
          월별
        </div>
      ),
    },
  ];

  return (
    <div className="Wrapper" style={{ padding: "1% 5%" }}>
      <div
        className="statHeader"
        style={{ width: "100%", textAlign: "center" }}
      >
        <h1>{`${new Date().getFullYear()}년 ${
          new Date().getMonth() + 1
        }월 ${new Date().getDate()}일 ${new Date().getHours()}:00 기준 통계`}</h1>
      </div>
      <div
        className="squareWrapper"
        style={{
          width: "100%",
          display: "flex",
          marginRight: "3%",
          marginTop: "5%",
          marginBottom: "5%",
        }}
      >
        {squareData.map((data, index) => {
          return (
            <div
              key={`sqaureData${index}`}
              style={{
                width: "25%",
                display: "flex",
                justifyContent: "center",
                flexDirection: "column",
              }}
            >
              <div
                className="squareHeader"
                style={{
                  width: "100%",
                  textAlign: "center",
                  fontWeight: "bold",
                  marginBottom: "3%",
                }}
              >
                {data.title}
              </div>
              <div
                className="squareBody"
                style={{
                  width: "80%",
                  height: "100%",
                  textAlign: "center",
                  fontSize: "30px",
                  border: "1px solid black",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  margin: "0 auto",
                }}
              >
                {data.dataObject}
              </div>
            </div>
          );
        })}
      </div>
      <Divider style={{ border: "1px solid black" }} />
      <div
        className="doughnutWrapper"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "20px",
        }}
      >
        {doughnutData.map((data, index) => {
          return (
            <div style={{ width: "300px" }} key={`doughnut${index}}`}>
              <DougnutGraph
                dataObject={data.dataObject}
                labelName={data.labelName}
                graphTitle={data.graphTitle}
              />
            </div>
          );
        })}
      </div>
      <Divider style={{ border: "1px solid black" }} />
      <div
        className="lineHeader"
        style={{
          width: "100%",
          display: "flex",
          justifyContent: "center",
        }}
      >
        <div style={{ marginRight: "2%" }}>
          {selectedType == "월별" ? (
            <DatePicker onChange={handleChangeDebut} picker="month" />
          ) : (
            <RangePicker
              onChange={handleChangeDebut}
              defaultValue={[
                dayjs(today.subtract(6, "day"), dateFormat),
                dayjs(today, dateFormat),
              ]}
              format={dateFormat}
            />
          )}
        </div>
        <div>
          <Dropdown menu={{ items }}>
            <Button>{selectedType}</Button>
          </Dropdown>
        </div>
      </div>
      {
        <div
          className="lineWrapper"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(2, 1fr)",
            gap: "20px",
          }}
        >
          {lineData.map((data, index) => {
            return (
              <div style={{ width: "500px" }} key={`line${index}`}>
                <LineGraph
                  dataObject={data.dataObject}
                  labelName={data.labelName}
                  graphTitle={data.graphTitle}
                />
              </div>
            );
          })}
        </div>
      }
    </div>
  );
};

export default AdminUserStat;
