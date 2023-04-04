// Auto close html details tag (add time series graph <details>) when the user clicks outside the tag

document.addEventListener('click', function (e) {
    let addTimeSeries = document.querySelector('details#add-time-series-details');

    if (!addTimeSeries.contains(e.target) || e.target.getAttribute('class') === 'add-time-series-option') {
        addTimeSeries.removeAttribute('open');
    }
})
