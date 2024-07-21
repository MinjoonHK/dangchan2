import { Chart as ChartJS, ArcElement, Tooltip, Legend, Title } from "chart.js";
import { Doughnut } from "react-chartjs-2";
import React from "react";
ChartJS.register(ArcElement, Tooltip, Legend, Title);

const DougnutGraph = ({ dataObject, labelName, graphTitle }) => {
  const options = {
    plugins: {
      legend: {
        position: "right" as const,
      },
      title: {
        display: true,
        text: graphTitle,
        font: {
          size: 20,
        },
      },
    },
  };
  const data = {
    labels: Object.keys(dataObject),

    datasets: [
      {
        label: labelName,
        data: Object.values(dataObject),
        backgroundColor: [
          "rgba(255, 99, 132, 0.2)",
          "rgba(54, 162, 235, 0.2)",
          "rgba(255, 206, 86, 0.2)",
          "rgba(75, 192, 192, 0.2)",
          "rgba(153, 102, 255, 0.2)",
          "rgba(255, 159, 64, 0.2)",
        ],
        borderColor: [
          "rgba(255, 99, 132, 1)",
          "rgba(54, 162, 235, 1)",
          "rgba(255, 206, 86, 1)",
          "rgba(75, 192, 192, 1)",
          "rgba(153, 102, 255, 1)",
          "rgba(255, 159, 64, 1)",
        ],
        borderWidth: 1,
      },
    ],
  };
  return (
    <div>
      <Doughnut options={options} data={data} />
    </div>
  );
};

export default DougnutGraph;
