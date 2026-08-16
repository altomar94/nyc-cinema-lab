THEATER_TICKET_URLS = {
    "Film Forum": "https://filmforum.org/now_playing",
    "IFC Center": "https://www.ifccenter.com/",
    "Metrograph": "https://metrograph.com/nyc/",
    "The Paris Theater": "https://www.paristheaternyc.com/",
    "The Roxy Cinema": "https://www.roxycinematribeca.com/",
    "Film at Lincoln Center": "https://www.filmlinc.org/now-playing/",
    "BAM Rose Cinemas": "https://www.bam.org/film",
    "AMC Lincoln Square 13": "https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13",
    "Regal Times Square": "https://www.regmovies.com/theatres/regal-e-walk-times-square"
}

# Add curated 70mm / special event screenings for these venues
FALLBACK_SCREENINGS.extend([
    create_entry(
        title="Oppenheimer",
        director="Christopher Nolan",
        year=2023,
        theater="AMC Lincoln Square 13",
        neighborhood="Upper West Side",
        summary="A biographical drama detailing the life of theoretical physicist J. Robert Oppenheimer, director of the Los Alamos Laboratory during the Manhattan Project.",
        fmt="70mm IMAX",
        showtimes=[f"Fri {fri_str}: 6:45 PM", f"Sat {sat_str}: 2:30 PM", f"Sun {sun_str}: 7:15 PM"],
        ticket_url="https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"
    ),
    create_entry(
        title="Dune: Part Two",
        director="Denis Villeneuve",
        year=2024,
        theater="AMC Lincoln Square 13",
        neighborhood="Upper West Side",
        summary="Paul Atreides unites with the Fremen people of the desert planet Arrakis to wage war against House Harkonnen.",
        fmt="IMAX Laser GT",
        showtimes=[f"Fri {fri_str}: 8:00 PM", f"Sat {sat_str}: 4:00 PM"],
        ticket_url="https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"
    ),
    create_entry(
        title="Heat",
        director="Michael Mann",
        year=1995,
        theater="Regal Times Square",
        neighborhood="Times Square",
        summary="A methodical thief and a relentless LAPD homicide detective engage in a lethal cat-and-mouse confrontation across Los Angeles.",
        fmt="4K Laser RPX",
        showtimes=[f"Sat {sat_str}: 8:30 PM", f"Sun {sun_str}: 5:15 PM"],
        ticket_url="https://www.regmovies.com/theatres/regal-e-walk-times-square"
    )
])
