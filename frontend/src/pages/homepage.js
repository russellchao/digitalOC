import React, {useRef, useEffect, useState} from 'react';
import TeamDropdownMenu from '../components/team_dropdown';

const Homepage = () => {

    /* 
        ---- Attributes for game situation ----
    
        NOTE: Some attributes will be directly inputted from the UI while other attributes will be calculated based on the 
        values of the inputted attributes (attributes for the play-by-play situation ML model)
    */
    
    // Team attributes
    const [offenseTeam, setOffenseTeam] = useState("");
    const [defenseTeam, setDefenseTeam] = useState("");

    // Down and distance attributes
    const [down, setDown] = useState();
    const [ydsToGo, setYdsToGo] = useState();
    const [ownOppMidfield, setOwnOppMidfield] = useState(""); // Defines whether the offense is in their own or opponent's territory or midfield
    const [ydLine50, setYdLine50] = useState(); // Yard line relative to own/opp territory (ignored if midfield selected)
    const [ydLine100, setYdLine100] = useState(); // Total distance to opponent's end zone
    const [goalToGo, setGoalToGo] = useState(false);

    // Score attributes
    const [offensePoints, setOffensePoints] = useState(); 
    const [defensePoints, setDefensePoints] = useState();
    const [scoreDiff, setScoreDiff] = useState();

    // Time attributes
    /* If OT is selected for quarter, the qtr/half/game seconds will be calculated as if it was the 4th quarter */
    const [quarter, setQuarter] = useState(""); 
    const [minutes, setMinutes] = useState(); 
    const [seconds, setSeconds] = useState(); 
    const [qtrSeconds, setQtrSeconds] = useState(); 
    const [halfSeconds, setHalfSeconds] = useState(); 
    const [gameSeconds, setGameSeconds] = useState(); 

    return (
        <div>
            {/* ---- Header Section ---- */}
            <div style={{ paddingLeft: "20px", paddingTop: "20px" }}>
                <h1>DigitalOC</h1>
            </div>

            <br />

            {/* ---- Team Input Section ---- */}
            <div style={{ paddingLeft: "60px", paddingTop: "20px" }}>
                <h2>Teams</h2>

                <TeamDropdownMenu 
                    defaultName="Offense" 
                    onChange={(value) => setOffenseTeam(value)}
                /> 
                <span style={{ paddingLeft: '20px' }}>vs.</span> 
                <TeamDropdownMenu 
                    defaultName="Defense" 
                    onChange={(value) => setDefenseTeam(value)}
                />
            </div>


            {/* ---- Down and Distance Section ---- */}
            <div style={{ paddingLeft: "60px", paddingTop: "20px" }}>
                <h2>Down & Distance</h2>

                <div style={{ paddingLeft: '20px' }}>
                    {/* Down Selector */}
                    <select
                        name="down"
                        id="down"
                        style={{
                            marginLeft: '5px',
                            padding: '5px',
                            fontSize: '16px',
                        }}
                        value={down}
                        onChange={(e) => setDown(e.target.value)}
                    >
                        <option value="" style={{ color: "gray" }}>---- Select Down ----</option>
                        <option value="1">1st</option>
                        <option value="2">2nd</option>
                        <option value="3">3rd</option>
                        <option value="4">4th</option>
                    </select>

                    <span style={{ paddingLeft: '10px' }}>&</span> 

                    {/* Yards to Go */}
                    <input
                        type="number"
                        name="ydsToGo"
                        id="ydsToGo"
                        placeholder="Yards to Go"
                        min="1"
                        max="99"
                        style={{
                            marginLeft: '10px',
                            padding: '5px',
                            fontSize: '16px',
                            width: '150px',
                        }}
                        value={ydsToGo}
                        onChange={(e) => {
                            const value = parseInt(e.target.value);
                            if ((value >= 1 && value <= 99) || e.target.value === '') {
                                setYdsToGo(e.target.value);
                            }
                        }}
                    />

                    <span style={{ paddingLeft: '10px' }}>from the</span> 

                    {/* Yard line */ }
                    <select
                        name="ownOppMidfield"
                        id="ownOppMidfield"
                        style={{
                            marginLeft: '5px',
                            padding: '5px',
                            fontSize: '16px',
                        }}
                        value={ownOppMidfield}
                        onChange={(e) => setOwnOppMidfield(e.target.value)}
                    >
                        <option value="" style={{ color: "gray" }}>---- Select Own/Opp/Midfield ----</option>
                        <option value="own">OWN</option>
                        <option value="opp">OPP</option>
                        <option value="midfield">MIDFIELD (50)</option>
                    </select>

                    <input
                        type="number"
                        name="ydLine50"
                        id="ydLine50"
                        placeholder="Yard Line"
                        min="1"
                        max="49"
                        style={{
                            marginLeft: '10px',
                            padding: '5px',
                            fontSize: '16px',
                            width: '100px',
                        }}
                        value={ydLine50}
                        onChange={(e) => {
                            const value = parseInt(e.target.value);
                            if ((value >= 1 && value <= 49) || e.target.value === '') {
                                setYdLine50(e.target.value);
                            }
                        }}
                    />
                </div>
            </div>

            {/* ---- Score Section ---- */}
            <div style={{ paddingLeft: "60px", paddingTop: "20px" }}>
                <h2>Score</h2>

                <div style={{ paddingLeft: '20px' }}>
                    {/* Offense Score */}
                    <input
                        type="number"
                        name="offensePoints"
                        id="offensePoints"
                        placeholder="Offense Points"
                        min="0"
                        style={{
                            marginLeft: '10px',
                            padding: '5px',
                            fontSize: '16px',
                            width: '150px',
                        }}
                        value={offensePoints}
                        onChange={(e) => {
                            const value = parseInt(e.target.value);
                            if ((value >= 0 && value <= 99) || e.target.value === '') {
                                setOffensePoints(e.target.value);
                            }
                        }}
                    />

                    <span style={{ paddingLeft: '10px' }}>-</span> 

                    {/* Defense Score */}
                    <input
                        type="number"
                        name="defensePoints"
                        id="defensePoints"
                        placeholder="Defense Points"
                        min="0"
                        style={{
                            marginLeft: '10px',
                            padding: '5px',
                            fontSize: '16px',
                            width: '150px',
                        }}
                        value={defensePoints}
                        onChange={(e) => {
                            const value = parseInt(e.target.value);
                            if ((value >= 0 && value <= 99) || e.target.value === '') {
                                setDefensePoints(e.target.value);
                            }
                        }}
                    />
                </div>
            </div>

            {/* ---- Time Section ---- */}


            
        </div>
    );
}


export default Homepage;
