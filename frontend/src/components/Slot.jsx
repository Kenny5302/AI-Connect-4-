import redToken from '../images/red_token.svg';
import blackToken from '../images/black_token.svg';

const Slot = ({ch, y, x}) => {
    return (
        <div className="slot" x={x} y={y}>
            {ch && (
                /*red token for player, black token for computer */
                <img src={ch === 'X' ? redToken : blackToken}
                width='100%' height='100%'/>
            )}

        </div>
    );
};

/* 'X' red,  'O' black */
export default Slot