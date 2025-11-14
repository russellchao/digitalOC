import React, {useRef, useEffect, useState} from 'react';

const Home = () => {
    return (
        <div style={{ textAlign: 'center', color: 'white', paddingTop: '20vh' }}>
            <h1>Welcome to DigitalOC</h1>
            <h4>
                An app that uses play-by-play data and team playcalling tendencies to train an ML model
                that suggests the most optimal offensive play in any NFL game situation.
            </h4>

            <button onClick={() => {
                window.location.href = '/situation';
            }}>
                Start
            </button>
        </div>
    );
}

export default Home;