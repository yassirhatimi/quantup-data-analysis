import unittest
import pandas as pd
from gold_waves import confirmed_pivots, impulse_events, pyramid_analysis, clean_bars, observation_times


class TestWaves(unittest.TestCase):
    def events(self,p):
        return impulse_events(p,pd.date_range('2025-01-01',periods=len(p),freq='h',tz='UTC'))

    def test_user_up_example(self):
        events=self.events([4300,4360,4450,4600,4650,4550,4500,4660])
        self.assertEqual(len(events),1)
        e=events[0]
        self.assertEqual(e['outcome'],'continued')
        self.assertEqual(e['move_before_pullback_usd'],350)
        self.assertEqual(e['max_pullback_usd'],150)

    def test_mirrored_down_example(self):
        e=self.events([4600,4540,4450,4300,4250,4350,4400,4240])[0]
        self.assertEqual(e['direction'],'down')
        self.assertEqual(e['outcome'],'continued')
        self.assertEqual(e['max_pullback_usd'],150)

    def test_full_retrace(self):
        e=self.events([4300,4360,4600,4500,4300])[0]
        self.assertEqual(e['outcome'],'fully_retraced')

    def test_unfinished_not_failure(self):
        e=self.events([4300,4360,4600,4650])[0]
        self.assertEqual(e['outcome'],'end_of_data')
        self.assertIsNone(e['pullback_utc'])

    def test_equal_high_not_breakout(self):
        e=self.events([4300,4360,4600,4500,4600])[0]
        self.assertEqual(e['outcome'],'end_of_data')

    def test_no_future_pivot_leak(self):
        p=confirmed_pivots([4300,4400,4600,4580],50)
        self.assertEqual(len(p),1)
        p=confirmed_pivots([4300,4400,4600,4580,4540],50)
        self.assertEqual(p[-1]['pivot_index'],2)
        self.assertEqual(p[-1]['confirmed_index'],4)

    def test_window_expiry(self):
        t=pd.to_datetime(['2025-01-01','2025-01-02','2025-01-03','2025-04-01'],utc=True)
        e=impulse_events([4300,4360,4600,4500],t)[0]
        self.assertEqual(e['outcome'],'window_expired')
        self.assertIsNone(e['pullback_utc'])

    def test_pyramid_asof_excludes_future_bars(self):
        f=pd.DataFrame({'close':[4300,4400,4600,4500,4650]},index=pd.date_range('2025-01-01',periods=5,freq='h',tz='UTC'))
        a=pyramid_analysis({'1h':f},as_of='2025-01-01T03:00:00Z')
        self.assertEqual(a.iloc[0].last_close,4600)

    def test_filter_no_interpolation(self):
        idx=pd.date_range('2025-01-01',periods=3,freq='h',tz='UTC')
        f=pd.DataFrame({'Open':[4300,4300,4300],'High':[4310,4300,4310],
                        'Low':[4290,4300,4290],'Close':[4305,4300,4306],
                        'Volume':[1,0,1]},index=idx)
        d,q=clean_bars(f,'1h',pd.Timestamp('2025-01-01T02:30:00Z'))
        self.assertEqual(q['usable_rows'],1)
        self.assertEqual(q['flat_zero_volume'],1)
        self.assertEqual(q['incomplete_bars'],1)

    def test_bad_parameters(self):
        with self.assertRaises(ValueError): confirmed_pivots([1,2],0)
        with self.assertRaises(ValueError): impulse_events([1,2],pd.date_range('2025-01-01',periods=2),impulse=25)

    def test_confirmation_uses_bar_end(self):
        f=pd.DataFrame({'close':[4300,4400,4600,4500]},index=pd.date_range('2025-01-01',periods=4,freq='h',tz='UTC'))
        p=pyramid_analysis({'1h':f})
        self.assertEqual(p.iloc[0].high_known_utc,'2025-01-01T04:00:00+00:00')
        self.assertEqual(observation_times(f,'15m')[0],pd.Timestamp('2025-01-01T00:15:00Z'))

    def test_pyramid_reports_broken_low(self):
        f=pd.DataFrame({'close':[4300,4400,4350,4450,4340]},index=pd.date_range('2025-01-01',periods=5,freq='h',tz='UTC'))
        p=pyramid_analysis({'1h':f})
        self.assertEqual(p.iloc[0].structure_break,'previous_confirmed_low_broken')


if __name__=='__main__': unittest.main()
