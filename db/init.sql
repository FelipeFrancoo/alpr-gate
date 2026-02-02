-- license plate is 10, even though no state that I know of has license plates longer than 8 characters, 
-- just to give it some wiggle room
CREATE TABLE IF NOT EXISTS main_gate_alpr_license_plates (
    -- filled out by the server
    id UUID PRIMARY KEY,
    license_plate VARCHAR(10) NOT NULL,
    captured_at TIMESTAMP NOT NULL,

    -- filled out by the user
    license_plate_corrected VARCHAR(10),
    visitor_name VARCHAR(50),
    visitor_company_name VARCHAR(50),
    visitor_receiver_name VARCHAR(50),
    visit_reason VARCHAR(200),
    soft_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    drove_away_at TIMESTAMP
);