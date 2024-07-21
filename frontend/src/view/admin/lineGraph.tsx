import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import React from "react";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function LineGraph({ dataObject, labelName, graphTitle }) {
  const data = {
    labels: Object.keys(dataObject),
    datasets: [
      {
        label: labelName,
        data: Object.values(dataObject),
        borderColor: "#85DAD2",
        backgroundColor: "white",
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: true,
        text: graphTitle,
      },
    },
  };

  return <Line options={options} data={data} />;
}

export default LineGraph;
