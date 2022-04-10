// alert('Hello World')


function renderDataInTheTable(results){

    console.log(JSON.parse(results))
    console.log(typeof(JSON.parse(results)))
    results = JSON.parse(results)

    const mytable = document.getElementById("predictions-table");
    results.forEach(element => {
        let newRow = document.createElement("tr");
        Object.values(element).forEach((value) => {
            let cell = document.createElement("td");
            cell.innerText=value;
            newRow.append(cell);
        });
        mytable.appendChild(newRow);
    });
}