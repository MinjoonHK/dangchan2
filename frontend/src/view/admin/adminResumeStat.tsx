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

const AdminResumeStat = () => {
  const [companyRanking, setCompanyRanking] = useState({});
  const [occupationRanking, setOccupationRanking] = useState({});
  const [trafficData, setTrafficData] = useState({});
  const [signupData, setSignupData] = useState({});
  const [selectedType, setSelectedType] = useState("타입");
  const [editTotal, setEditTotal] = useState(0);
  const [editToday, setEditToday] = useState(0);
  const [totalResume, setTotalResume] = useState(0);
  const [modeRatio, setModeRatio] = useState({});
  const [statusRatio, setStatusRatio] = useState({});
  const [groupRatio, setGroupRatio] = useState({});
  const [dailyData, setDailyData] = useState({});
  const [monthlyData, setMonthlyData] = useState({});

  const fetchSquareData = async (group: string) => {
    let res = await axiosInstance
      .get(`statistics/resume/count/${group}`)
      .then((res) => {
        switch (group) {
          case "total-edit":
            setEditTotal(res.data.total_edit);
            break;
          case "today-edit":
            setEditToday(res.data.today_edit);
            break;
          case "toal-board":
            setTotalResume(res.data.total_board);
            break;
        }
        console.log(res);
      })
      .catch((err) => {
        console.log("박스데이터를 가져오는데 오류가 발생했습니다", err);
      });
  };

  const fetchRankData = (group: string) => {
    let res = axiosInstance
      .get(`admin/statistics/rank/${group}`)
      .then((res) => {
        switch (group) {
          case "occupation":
            setOccupationRanking(res.data.response.ranking_resumeEdit);
            break;
          case "company":
            setCompanyRanking(res.data.response.ranking_resumeEdit);
            break;
        }
      })
      .catch((err) => {
        console.log("랭킹데이터를 가져오는데 오류가 발생했습니다", err);
      });
  };

  const fetchRatioData = (group: string) => {
    let res = axiosInstance
      .get("admin/statistics/resume-edit", {
        params: {
          group: group,
        },
      })
      .then((res) => {
        switch (group) {
          case "mode":
            let modeData = res.data.response.edit_cnt;
            let newModeData = {
              PRO: modeData.mode_pro,
              LITE: modeData.mode_light,
            };
            setModeRatio(newModeData);
            break;
          case "status":
            let statusData = res.data.response.edit_cnt;
            let newStatusData = {
              구직자: statusData.status_1,
              이직자: statusData.status_2,
            };
            setStatusRatio(newStatusData);
            break;
          case "age":
            setGroupRatio(res.data.response.age_edit_cnt);
            break;
        }
      });
  };
  const fetchLineData = async (group: string) => {
    try {
      const res = await axiosInstance
        .get(`/admin/statistics/resume/${group}`)
        .then((res) => {
          switch (group) {
            case "daily":
              setDailyData(res.data.response.traffic_data);
              break;
            case "monthly":
              setMonthlyData(res.data.response.signup_data);
              break;
          }
        });
    } catch (err) {
      console.log("라인데이터를 불러오는데에 실패했습니다!", err);
    }
  };

  const fetchAccumulatedData = async (group: string) => {
    try {
      const res = await axiosInstance.get(`/admin/statistics/user/${group}`);
      switch (group) {
        case "traffic":
          setTrafficData(res.data.response.traffic_data);
          break;
        case "signup":
          setSignupData(res.data.response.signup_data);
          break;
      }
    } catch (err) {
      console.log("집계데이터를 불러오는데에 실패했습니다!", err);
    }
  };

  const fetchData = async () => {
    await Promise.all([
      fetchAccumulatedData("traffic"),
      fetchAccumulatedData("signup"),
      fetchSquareData("total-edit"),
      fetchSquareData("today-edit"),
      fetchSquareData("total-board"),
      fetchRankData("occupation"),
      fetchRankData("company"),
      fetchRatioData("mode"),
      fetchRatioData("status"),
      fetchRatioData("age"),
      fetchLineData("daily"),
      fetchLineData("monthly"),
    ]);
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleChangeDebut = (range) => {
    if (selectedType == "월별") {
      const valueOfInput = range.format();
      const res = axiosInstance
        .get("/admin/statistics/user/access", {
          params: {
            month: valueOfInput.slice(0, 10),
          },
        })
        .then((res) => {
          setTrafficData(res.data.response.traffic_data);
        });
      const res2 = axiosInstance
        .get("/admin/statistics/user/signup", {
          params: {
            month: valueOfInput.slice(0, 10),
          },
        })
        .then((res) => {
          setSignupData(res.data.response.signup_data);
        });
    } else {
      const valueOfInput1 = range[0].format();
      const valueOfInput2 = range[1].format();
      const res = axiosInstance
        .get("/admin/statistics/user/access", {
          params: {
            startDate: valueOfInput1.slice(0, 10),
            endDate: valueOfInput2.slice(0, 10),
          },
        })
        .then((res) => {
          setTrafficData(res.data.response.traffic_data);
        });
      const res2 = axiosInstance
        .get("/admin/statistics/user/signup", {
          params: {
            startDate: valueOfInput1.slice(0, 10),
            endDate: valueOfInput2.slice(0, 10),
          },
        })
        .then((res) => {
          setSignupData(res.data.response.signup_data);
        });
    }
  };

  const doughnutData = [
    {
      dataObject: modeRatio,
      labelName: "모드",
      graphTitle: "PRO/LITE 첨삭 비율",
    },
    {
      dataObject: statusRatio,
      labelName: "유저 상태",
      graphTitle: "구직자/이직자 첨삭 비율",
    },
    {
      dataObject: groupRatio,
      labelName: "나잇대",
      graphTitle: "연령별 첨삭 비율",
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
  ];

  const squareData = [
    {
      dataObject: editTotal,
      title: "총 첨삭수",
    },
    {
      dataObject: editToday,
      title: "오늘 첨삭수",
    },
    {
      dataObject: totalResume,
      title: "총 자소서 게시글",
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
                width: "35%",
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
            <DatePicker
              onChange={handleChangeDebut}
              picker="month"
              defaultValue={dayjs("2024/05")}
            />
          ) : (
            <RangePicker
              onChange={handleChangeDebut}
              defaultValue={[
                dayjs("2024/05/01", dateFormat),
                dayjs("2024/05/07", dateFormat),
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

export default AdminResumeStat;
